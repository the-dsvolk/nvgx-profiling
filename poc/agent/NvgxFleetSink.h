// SPDX-License-Identifier: Apache-2.0
// nvgx FleetSink (POC): a table sink for NVIDIA's nvtrace_continuous_export that turns each snapshot into a
// compact workload profile (JSON) and POSTs it to a fleet API from a background thread.
//
// It implements the same 7-method sink interface as nv::trace::DuckDbTableSink, and TeeSink feeds both.
// Column positions follow NvTraceTableSchema.h of the Nsight Perf SDK for AI beta-preview-update1.
// See design/fleet-profiling.md in https://github.com/the-dsvolk/nvgx-profiling.
#pragma once

#include <NvTrace/Offline/Exporter/NvTraceTableSchema.h>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace nvgx {

    using nv::trace::TableValue;

    // ---------- value helpers ----------
    inline std::optional<uint64_t> U64(const TableValue* v, size_t n, size_t i)
    {
        if (i >= n)
            return std::nullopt;
        return std::visit(
            [](auto&& x) -> std::optional<uint64_t> {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, bool>)
                    return x ? 1 : 0;
                else if constexpr (std::is_integral_v<T>)
                    return static_cast<uint64_t>(x);
                else if constexpr (std::is_same_v<T, double>)
                    return static_cast<uint64_t>(x);
                else
                    return std::nullopt;
            },
            v[i]);
    }
    inline std::optional<double> Dbl(const TableValue* v, size_t n, size_t i)
    {
        if (i >= n)
            return std::nullopt;
        if (const double* d = std::get_if<double>(&v[i]))
            return *d;
        if (auto u = U64(v, n, i))
            return static_cast<double>(*u);
        return std::nullopt;
    }
    inline std::string_view Str(const TableValue* v, size_t n, size_t i)
    {
        if (i >= n)
            return {};
        if (const std::string_view* s = std::get_if<std::string_view>(&v[i]))
            return *s;
        return {};
    }

    struct PairHash
    {
        size_t operator()(const std::pair<uint64_t, uint64_t>& p) const
        {
            return std::hash<uint64_t>()(p.first * 0x9E3779B97F4A7C15ULL ^ p.second);
        }
    };
    using Key = std::pair<uint64_t, uint64_t>;

    // ---------- kernel families (fixed order: drives dashboard colors) ----------
    enum Family : int { Gemm, FlashAttention, PytorchNative, FlashInferSampling, VllmCustom, Other, Unresolved, FamilyCount };
    inline const char* FamilyName(int f)
    {
        static const char* names[] = {"gemm", "flash_attention", "pytorch_native", "flashinfer_sampling", "vllm_custom", "other", "unresolved"};
        return names[f];
    }
    inline int Classify(std::string_view n)
    {
        auto has = [&](std::string_view s) { return n.find(s) != std::string_view::npos; };
        if (n.empty() || has("??"))
            return Unresolved;
        if (has("flash_fwd"))
            return FlashAttention;
        if (has("cutlass") || n.rfind("nvjet", 0) == 0)
            return Gemm;
        if (has("at::native"))
            return PytorchNative;
        if (has("flashinfer"))
            return FlashInferSampling;
        if (has("vllm::"))
            return VllmCustom;
        return Other;
    }
    // "void flash::flash_fwd_splitkv_kernel<...>(...)" -> "flash::flash_fwd_splitkv_kernel"
    inline std::string ShortName(std::string_view n)
    {
        if (n.rfind("void ", 0) == 0)
            n.remove_prefix(5);
        std::string out;
        int depth = 0;
        for (char c : n)
        {
            if (c == '<' || c == '(')
            {
                if (depth == 0 && !out.empty())
                    break;
                ++depth;
            }
            else if (c == '>' || c == ')')
                depth = std::max(0, depth - 1);
            else if (depth == 0)
                out.push_back(c);
        }
        while (!out.empty() && out.back() == ' ')
            out.pop_back();
        if (out.empty())
            out = std::string(n.substr(0, 80));
        return out.substr(0, 80);
    }

    // ---------- minimal JSON writer ----------
    class Json
    {
    public:
        Json& Raw(std::string_view s) { m_ss << s; return *this; }
        Json& Str(std::string_view s)
        {
            m_ss << '"';
            for (unsigned char c : s)
            {
                switch (c)
                {
                case '"': m_ss << "\\\""; break;
                case '\\': m_ss << "\\\\"; break;
                case '\n': m_ss << "\\n"; break;
                case '\r': m_ss << "\\r"; break;
                case '\t': m_ss << "\\t"; break;
                default:
                    if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); m_ss << b; }
                    else m_ss << c;
                }
            }
            m_ss << '"';
            return *this;
        }
        Json& Key(std::string_view k) { Comma(); Str(k); m_ss << ':'; m_needComma = false; return *this; }
        Json& Num(double v, int decimals = 2)
        {
            Comma();
            char b[64];
            std::snprintf(b, sizeof(b), "%.*f", decimals, v);
            m_ss << b;
            m_needComma = true;
            return *this;
        }
        Json& Int(uint64_t v) { Comma(); m_ss << v; m_needComma = true; return *this; }
        Json& Null() { Comma(); m_ss << "null"; m_needComma = true; return *this; }
        Json& Text(std::string_view s) { Comma(); Str(s); m_needComma = true; return *this; }
        Json& Begin(char c) { Comma(); m_ss << c; m_needComma = false; return *this; }
        Json& End(char c) { m_ss << c; m_needComma = true; return *this; }
        std::string Take() { return m_ss.str(); }

    private:
        void Comma() { if (m_needComma) m_ss << ','; }
        std::ostringstream m_ss;
        bool m_needComma = false;
    };

    inline double Percentile(std::vector<double>& v, double p)
    {
        if (v.empty())
            return 0;
        size_t k = static_cast<size_t>((v.size() - 1) * p / 100.0);
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    }

    // ---------- background HTTP sender ----------
    class Sender
    {
    public:
        bool Start(const std::string& url)
        {
            // http://host:port[/base]
            const std::string prefix = "http://";
            if (url.rfind(prefix, 0) != 0)
            {
                std::cerr << "[fleet] only http:// URLs are supported: " << url << '\n';
                return false;
            }
            std::string rest = url.substr(prefix.size());
            size_t slash = rest.find('/');
            std::string hostPort = rest.substr(0, slash);
            m_base = slash == std::string::npos ? "" : rest.substr(slash);
            while (!m_base.empty() && m_base.back() == '/')
                m_base.pop_back();
            size_t colon = hostPort.rfind(':');
            m_host = colon == std::string::npos ? hostPort : hostPort.substr(0, colon);
            m_port = colon == std::string::npos ? "80" : hostPort.substr(colon + 1);
            m_thread = std::thread([this] { Run(); });
            return true;
        }
        void Stop()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stop = true;
            }
            m_cv.notify_all();
            if (m_thread.joinable())
                m_thread.join();
        }
        void Put(std::string profileJson)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queue.size() >= kMaxQueue)
                m_queue.pop_front(); // bounded memory: drop the oldest
            m_queue.push_back(std::move(profileJson));
            m_cv.notify_one();
        }
        size_t Pending()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }

    private:
        static constexpr size_t kMaxQueue = 720;

        void Run()
        {
            int backoffSec = 1;
            for (;;)
            {
                std::vector<std::string> batch;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_cv.wait(lock, [this] { return m_stop || !m_queue.empty(); });
                    if (m_queue.empty())
                        return; // stop requested and nothing left
                    for (size_t i = 0; i < m_queue.size() && i < 20; ++i)
                        batch.push_back(m_queue[i]);
                }
                std::string body = "[";
                for (size_t i = 0; i < batch.size(); ++i)
                    body += (i ? "," : "") + batch[i];
                body += "]";
                if (Post(body))
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    for (size_t i = 0; i < batch.size() && !m_queue.empty(); ++i)
                        m_queue.pop_front();
                    backoffSec = 1;
                }
                else
                {
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        if (m_stop)
                            return;
                    }
                    std::cerr << "[fleet] POST failed; retrying in " << backoffSec << " s (queued " << Pending() << ")\n";
                    std::this_thread::sleep_for(std::chrono::seconds(backoffSec));
                    backoffSec = std::min(backoffSec * 2, 30);
                }
            }
        }

        bool Post(const std::string& body)
        {
            addrinfo hints{}, *res = nullptr;
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            if (getaddrinfo(m_host.c_str(), m_port.c_str(), &hints, &res) != 0)
                return false;
            int fd = -1;
            for (addrinfo* ai = res; ai; ai = ai->ai_next)
            {
                fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0)
                    continue;
                timeval tv{5, 0};
                setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0)
                    break;
                close(fd);
                fd = -1;
            }
            freeaddrinfo(res);
            if (fd < 0)
                return false;
            std::ostringstream req;
            req << "POST " << m_base << "/v1/profiles HTTP/1.1\r\nHost: " << m_host << ':' << m_port
                << "\r\nContent-Type: application/json\r\nContent-Length: " << body.size()
                << "\r\nConnection: close\r\n\r\n" << body;
            const std::string data = req.str();
            size_t sent = 0;
            while (sent < data.size())
            {
                ssize_t k = send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
                if (k <= 0)
                {
                    close(fd);
                    return false;
                }
                sent += static_cast<size_t>(k);
            }
            char buf[64] = {};
            ssize_t k = recv(fd, buf, sizeof(buf) - 1, 0);
            close(fd);
            // "HTTP/1.x 2xx"
            return k >= 12 && std::strncmp(buf, "HTTP/1.", 7) == 0 && buf[9] == '2';
        }

        std::string m_host, m_port, m_base;
        std::thread m_thread;
        std::mutex m_mutex;
        std::condition_variable m_cv;
        std::deque<std::string> m_queue;
        bool m_stop = false;
    };

    // ---------- the sink ----------
    class FleetSink
    {
    public:
        bool Open(const std::string& url, const std::string& node)
        {
            m_node = node;
            m_enabled = m_sender.Start(url);
            if (m_enabled)
                std::cout << "[fleet] sending workload profiles to " << url << " as node " << node << ".\n";
            return m_enabled;
        }
        void Close()
        {
            if (m_enabled)
                m_sender.Stop();
            m_enabled = false;
        }

        bool BeginSnapshotIngestionTransaction()
        {
            m_snap = Snapshot{};
            m_snap.ingestStart = std::chrono::steady_clock::now();
            return true;
        }
        void AbortSnapshotIngestionTransaction() { m_snap = Snapshot{}; }

        bool AppendRow(size_t table, const TableValue* v, size_t n)
        {
            if (!m_enabled)
                return true;
            using namespace nv::trace;
            switch (table)
            {
            case TableIndex_SnapshotMetadata:
                m_snap.windowStart = U64(v, n, 2).value_or(0);
                m_snap.windowEnd = U64(v, n, 3).value_or(0);
                m_snap.snapshotNumber = U64(v, n, 4).value_or(0);
                break;
            case TableIndex_Processes:
                if (auto pid = U64(v, n, 5))
                    m_hostPid[U64(v, n, 2).value_or(0)] = static_cast<int32_t>(*pid);
                break;
            case TableIndex_CudaFunctions:
            {
                std::string_view name = Str(v, n, 7);
                m_functions[{U64(v, n, 2).value_or(0), U64(v, n, 3).value_or(0)}] = {Classify(name), ShortName(name)};
                break;
            }
            case TableIndex_CudaDriverApiCalls:
            {
                const uint64_t proc = U64(v, n, 2).value_or(0);
                auto start = U64(v, n, 5), end = U64(v, n, 6);
                m_snap.apiCalls++;
                m_snap.apiThreads.insert(U64(v, n, 3).value_or(0));
                if (start && end && *end > *start && Str(v, n, 8).find("Synchronize") != std::string_view::npos)
                    m_snap.syncNs += *end - *start;
                if (start)
                    m_apiStartNew[{proc, U64(v, n, 4).value_or(0)}] = *start;
                break;
            }
            case TableIndex_CudaGraphLaunches:
            {
                const uint64_t proc = U64(v, n, 2).value_or(0);
                auto apiStart = LookupApiStart(proc, U64(v, n, 11));
                if (auto ts = U64(v, n, 14))
                {
                    if (apiStart && *ts > *apiStart)
                        m_snap.delaysMs.push_back((*ts - *apiStart) / 1e6);
                }
                else if (apiStart)
                    m_graphPendingNew[{proc, U64(v, n, 8).value_or(0)}] = *apiStart;
                break;
            }
            case TableIndex_CudaImmediateKernelLaunches:
            {
                const uint64_t proc = U64(v, n, 2).value_or(0);
                const auto* fn = LookupFunction(proc, U64(v, n, 11).value_or(0));
                auto s = U64(v, n, 24), e = U64(v, n, 25);
                auto apiStart = LookupApiStart(proc, U64(v, n, 8));
                if (s && e)
                {
                    AddKernel(fn, *s, *e, false, proc);
                    if (apiStart && *s > *apiStart)
                        m_snap.delaysMs.push_back((*s - *apiStart) / 1e6);
                }
                else
                    m_kernelPendingNew[{proc, U64(v, n, 6).value_or(0)}] = {fn ? fn->family : int(Unresolved), fn ? fn->shortName : std::string("??"), apiStart.value_or(0), 0};
                break;
            }
            case TableIndex_CudaGraphNodeLaunches:
            {
                const uint64_t proc = U64(v, n, 2).value_or(0);
                const auto* fn = LookupFunction(proc, U64(v, n, 7).value_or(0));
                auto s = U64(v, n, 19), e = U64(v, n, 20);
                if (s && e)
                    AddKernel(fn, *s, *e, true, proc);
                else if (s)
                    m_nodePendingNew[{proc, U64(v, n, 5).value_or(0)}] = {fn ? fn->family : int(Unresolved), fn ? fn->shortName : std::string("??"), 0, *s};
                break;
            }
            case TableIndex_CudaGpuCommands:
            {
                auto s = U64(v, n, 10), e = U64(v, n, 11);
                if (s && e && *e > *s && Str(v, n, 15).rfind("Memcpy", 0) == 0)
                    m_snap.copyNs += *e - *s;
                break;
            }
            case TableIndex_CudaGpuMemcpyDetails:
            {
                const uint64_t bytes = U64(v, n, 11).value_or(0);
                std::string dir(Str(v, n, 21));
                std::transform(dir.begin(), dir.end(), dir.begin(), ::tolower);
                m_snap.copies++;
                if (bytes < 1024)
                    m_snap.tinyCopies++;
                if (dir.find("htod") != std::string::npos || dir.find("h2d") != std::string::npos)
                    m_snap.h2d += bytes;
                else if (dir.find("dtoh") != std::string::npos || dir.find("d2h") != std::string::npos)
                    m_snap.d2h += bytes;
                else if (dir.find("dtod") != std::string::npos || dir.find("d2d") != std::string::npos)
                    m_snap.d2d += bytes;
                break;
            }
            case TableIndex_GpuPartitions:
                m_chip = std::string(Str(v, n, 2));
                m_device = std::string(Str(v, n, 3));
                break;
            case TableIndex_GpuMetrics:
                if (auto val = Dbl(v, n, 3))
                    m_snap.metrics[std::string(Str(v, n, 2))].push_back(*val);
                break;
            default:
                break;
            }
            return true;
        }

        bool QueueUpdate(size_t update, const TableValue* v, size_t n)
        {
            if (!m_enabled)
                return true;
            using namespace nv::trace;
            if (update == TableUpdateIndex_CudaImmediateKernelLaunchGpuTimes)
            {
                Key k{U64(v, n, 2).value_or(0), U64(v, n, 4).value_or(0)};
                auto s = U64(v, n, 5), e = U64(v, n, 6);
                if (const Pending* p = FindPending(m_kernelPendingNew, m_kernelPendingPrev, k); p && s && e)
                {
                    AddKernelFamily(p->family, p->shortName, *s, *e, false, k.first);
                    if (p->apiStart && *s > p->apiStart)
                        m_snap.delaysMs.push_back((*s - p->apiStart) / 1e6);
                }
            }
            else if (update == TableUpdateIndex_CudaGraphNodeLaunchGpuEnd)
            {
                Key k{U64(v, n, 2).value_or(0), U64(v, n, 4).value_or(0)};
                auto e = U64(v, n, 6);
                if (const Pending* p = FindPending(m_nodePendingNew, m_nodePendingPrev, k); p && e)
                    AddKernelFamily(p->family, p->shortName, p->gpuStart, *e, true, k.first);
            }
            else if (update == TableUpdateIndex_CudaGraphLaunchTimestamp)
            {
                Key k{U64(v, n, 2).value_or(0), U64(v, n, 4).value_or(0)};
                auto ts = U64(v, n, 6);
                std::optional<uint64_t> apiStart;
                if (auto it = m_graphPendingPrev.find(k); it != m_graphPendingPrev.end())
                    apiStart = it->second;
                else if (auto it2 = m_graphPendingNew.find(k); it2 != m_graphPendingNew.end())
                    apiStart = it2->second;
                if (ts && apiStart && *ts > *apiStart)
                    m_snap.delaysMs.push_back((*ts - *apiStart) / 1e6);
            }
            return true;
        }

        bool CommitSnapshotIngestionTransaction()
        {
            if (!m_enabled)
                return true;
            const auto ingestMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_snap.ingestStart).count();
            m_sender.Put(BuildProfile(ingestMs));
            // Rows still missing their GPU times may complete in the next snapshot; keep one generation.
            m_kernelPendingPrev = std::move(m_kernelPendingNew);
            m_nodePendingPrev = std::move(m_nodePendingNew);
            m_graphPendingPrev = std::move(m_graphPendingNew);
            m_apiStartPrev = std::move(m_apiStartNew);
            m_kernelPendingNew.clear();
            m_nodePendingNew.clear();
            m_graphPendingNew.clear();
            m_apiStartNew.clear();
            if (!m_printedMetrics && !m_snap.metrics.empty())
            {
                std::cout << "[fleet] GPU metrics seen:";
                for (auto& [name, _] : m_snap.metrics)
                    std::cout << ' ' << name;
                std::cout << '\n';
                m_printedMetrics = true;
            }
            m_snap = Snapshot{};
            return true;
        }

    private:
        struct FuncInfo
        {
            int family = Unresolved;
            std::string shortName;
        };
        struct Pending
        {
            int family = Unresolved;
            std::string shortName;
            uint64_t apiStart = 0;
            uint64_t gpuStart = 0;
        };
        struct KernelStat
        {
            uint64_t count = 0, timeNs = 0;
        };
        struct Snapshot
        {
            std::chrono::steady_clock::time_point ingestStart;
            uint64_t windowStart = 0, windowEnd = 0, snapshotNumber = 0;
            uint64_t kernels = 0, kernelNs = 0, tiny = 0, graphKernels = 0;
            uint64_t familyNs[FamilyCount] = {};
            std::unordered_map<std::string, KernelStat> byName;
            uint64_t apiCalls = 0, syncNs = 0;
            std::set<uint64_t> apiThreads;
            std::set<uint64_t> processes;
            std::vector<double> delaysMs;
            uint64_t copyNs = 0, copies = 0, tinyCopies = 0, h2d = 0, d2h = 0, d2d = 0;
            std::map<std::string, std::vector<double>> metrics;
        };
        using PendingMap = std::unordered_map<Key, Pending, PairHash>;

        const FuncInfo* LookupFunction(uint64_t proc, uint64_t fn) const
        {
            auto it = m_functions.find({proc, fn});
            return it == m_functions.end() ? nullptr : &it->second;
        }
        std::optional<uint64_t> LookupApiStart(uint64_t proc, std::optional<uint64_t> corr) const
        {
            if (!corr)
                return std::nullopt;
            Key k{proc, *corr};
            if (auto it = m_apiStartNew.find(k); it != m_apiStartNew.end())
                return it->second;
            if (auto it = m_apiStartPrev.find(k); it != m_apiStartPrev.end())
                return it->second;
            return std::nullopt;
        }
        static const Pending* FindPending(const PendingMap& a, const PendingMap& b, const Key& k)
        {
            if (auto it = b.find(k); it != b.end())
                return &it->second;
            if (auto it = a.find(k); it != a.end())
                return &it->second;
            return nullptr;
        }
        void AddKernel(const FuncInfo* fn, uint64_t s, uint64_t e, bool graph, uint64_t proc)
        {
            AddKernelFamily(fn ? fn->family : int(Unresolved), fn ? fn->shortName : std::string("??"), s, e, graph, proc);
        }
        void AddKernelFamily(int family, const std::string& name, uint64_t s, uint64_t e, bool graph, uint64_t proc)
        {
            if (e <= s)
                return;
            const uint64_t d = e - s;
            m_snap.kernels++;
            m_snap.kernelNs += d;
            m_snap.familyNs[family] += d;
            if (d < 10'000)
                m_snap.tiny++;
            if (graph)
                m_snap.graphKernels++;
            auto& st = m_snap.byName[name];
            st.count++;
            st.timeNs += d;
            m_snap.processes.insert(proc);
        }

        static void Counter(Json& j, const char* key, std::map<std::string, std::vector<double>>& metrics, const std::vector<std::string>& needles)
        {
            for (auto& [name, vals] : metrics)
            {
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                bool all = true;
                for (auto& nd : needles)
                    all = all && lower.find(nd) != std::string::npos;
                if (!all || vals.empty())
                    continue;
                double sum = 0;
                for (double x : vals)
                    sum += x;
                std::vector<double> copy = vals;
                j.Key(key).Begin('{').Key("avg").Num(sum / vals.size()).Key("p99").Num(Percentile(copy, 99)).Key("metric").Text(name).End('}');
                return;
            }
        }

        std::string BuildProfile(double ingestMs)
        {
            Snapshot& s = m_snap;
            const uint64_t windowNs = s.windowEnd > s.windowStart ? s.windowEnd - s.windowStart : 1;
            const double secs = windowNs / 1e9;
            auto pct = [](double part, double whole) { return whole > 0 ? 100.0 * part / whole : 0.0; };
            const double busy = std::min(100.0, pct(double(s.kernelNs), double(windowNs)));
            const double sync = pct(double(s.syncNs), double(windowNs) * std::max<size_t>(1, s.apiThreads.size()));
            const double tiny = pct(double(s.tiny), double(s.kernels));
            const double copyPct = pct(double(s.copyNs), double(windowNs));

            Json j;
            j.Begin('{');
            j.Key("schema").Text("nvgx.workload_profile/v0");
            j.Key("node").Text(m_node);
            j.Key("snapshot_number").Int(s.snapshotNumber);
            j.Key("gpu").Begin('{').Key("index").Int(0).Key("chip").Text(m_chip).Key("device").Text(m_device).End('}');
            j.Key("processes").Begin('[');
            for (uint64_t p : s.processes)
            {
                auto it = m_hostPid.find(p);
                j.Int(it == m_hostPid.end() ? p : static_cast<uint64_t>(it->second));
            }
            j.End(']');
            j.Key("window").Begin('{').Key("start_utc_ns").Int(s.windowStart).Key("end_utc_ns").Int(s.windowEnd).Key("duration_s").Num(secs, 3).End('}');

            j.Key("attribution").Begin('{');
            j.Key("gpu_busy_pct").Num(busy, 1).Key("kernel_time_ms").Num(s.kernelNs / 1e6, 1);
            j.Key("by_kernel_family_ms").Begin('{');
            for (int f = 0; f < FamilyCount; ++f)
                j.Key(FamilyName(f)).Num(s.familyNs[f] / 1e6, 1);
            j.End('}').End('}');

            std::vector<std::pair<std::string, KernelStat>> top(s.byName.begin(), s.byName.end());
            std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second.timeNs > b.second.timeNs; });
            j.Key("kernels").Begin('{');
            j.Key("launches").Int(s.kernels).Key("launches_per_s").Num(s.kernels / secs, 1);
            j.Key("tiny_kernel_share_pct").Num(tiny, 1).Key("graph_share_pct").Num(pct(double(s.graphKernels), double(s.kernels)), 1);
            j.Key("top").Begin('[');
            for (size_t i = 0; i < top.size() && i < 5; ++i)
                j.Begin('{').Key("name").Text(top[i].first).Key("launches").Int(top[i].second.count).Key("time_ms").Num(top[i].second.timeNs / 1e6, 2).End('}');
            j.End(']').End('}');

            j.Key("launch_path").Begin('{');
            j.Key("api_calls_per_s").Num(s.apiCalls / secs, 1).Key("api_threads").Int(s.apiThreads.size());
            if (s.delaysMs.empty())
                j.Key("launch_to_start_ms_p50").Null().Key("launch_to_start_ms_p95").Null();
            else
            {
                std::vector<double> d = s.delaysMs;
                j.Key("launch_to_start_ms_p50").Num(Percentile(d, 50), 3);
                j.Key("launch_to_start_ms_p95").Num(Percentile(d, 95), 3);
            }
            j.Key("cpu_sync_wait_pct").Num(sync, 1).End('}');

            j.Key("memory").Begin('{');
            j.Key("h2d_bytes").Int(s.h2d).Key("d2h_bytes").Int(s.d2h).Key("d2d_bytes").Int(s.d2d);
            j.Key("copies").Int(s.copies).Key("tiny_copies").Int(s.tinyCopies).Key("copy_time_pct").Num(copyPct, 1).End('}');

            j.Key("counters").Begin('{');
            // Percent-of-peak metrics only (e.g. sm__inst_executed.sum is a raw count, not a utilization).
            Counter(j, "tensor_active_pct", s.metrics, {"tensor", "pct"});
            Counter(j, "sm_util_pct", s.metrics, {"sm__inst", "pct"});
            j.End('}');

            // Verdict heuristics (design/fleet-profiling.md section 7); raw numbers travel with the label.
            double sm = 0, tensor = 0;
            for (auto& [name, vals] : s.metrics)
            {
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("pct") == std::string::npos || vals.empty())
                    continue; // only percent-of-peak metrics feed the verdict
                double avg = 0;
                for (double x : vals)
                    avg += x / vals.size();
                if (lower.find("tensor") != std::string::npos)
                    tensor = avg;
                else if (lower.find("sm__inst") != std::string::npos)
                    sm = avg;
            }
            const char* verdict = busy < 10 ? "idle"
                : (sm >= 50 || tensor >= 50) ? "compute_bound"
                : copyPct >= 20 ? "copy_bound"
                : (sync >= 50 && tiny >= 40) ? "launch_sync_bound"
                : busy < 70 ? "host_bound" : "mixed";
            j.Key("verdict").Text(verdict);

            const uint64_t nowNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            j.Key("agent").Begin('{').Key("source").Text("nvtrace_continuous_export+FleetSink").Key("ingest_ms").Num(ingestMs, 1).Key("sent_utc_ns").Int(nowNs).Key("queued").Int(m_sender.Pending()).End('}');
            j.End('}');
            return j.Take();
        }

        bool m_enabled = false;
        bool m_printedMetrics = false;
        std::string m_node, m_chip, m_device;
        Sender m_sender;
        Snapshot m_snap;
        std::unordered_map<Key, FuncInfo, PairHash> m_functions;
        std::unordered_map<uint64_t, int32_t> m_hostPid;
        std::unordered_map<Key, uint64_t, PairHash> m_apiStartNew, m_apiStartPrev;
        PendingMap m_kernelPendingNew, m_kernelPendingPrev, m_nodePendingNew, m_nodePendingPrev;
        std::unordered_map<Key, uint64_t, PairHash> m_graphPendingNew, m_graphPendingPrev;
    };

    // Feeds every row to two sinks (e.g. DuckDB for local storage + FleetSink for the fleet).
    template <typename A, typename B>
    class TeeSink
    {
    public:
        TeeSink(A& a, B& b) : m_a(a), m_b(b) {}
        bool BeginSnapshotIngestionTransaction() { return m_a.BeginSnapshotIngestionTransaction() && m_b.BeginSnapshotIngestionTransaction(); }
        bool AppendRow(size_t t, const TableValue* v, size_t n) { const bool ok = m_a.AppendRow(t, v, n); m_b.AppendRow(t, v, n); return ok; }
        bool QueueUpdate(size_t u, const TableValue* v, size_t n) { const bool ok = m_a.QueueUpdate(u, v, n); m_b.QueueUpdate(u, v, n); return ok; }
        bool CommitSnapshotIngestionTransaction() { const bool ok = m_a.CommitSnapshotIngestionTransaction(); m_b.CommitSnapshotIngestionTransaction(); return ok; }
        void AbortSnapshotIngestionTransaction() { m_a.AbortSnapshotIngestionTransaction(); m_b.AbortSnapshotIngestionTransaction(); }

    private:
        A& m_a;
        B& m_b;
    };

} // namespace nvgx
