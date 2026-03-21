#include <mpi.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace tensorrt_llm::executor;

// ── CLI helpers ──────────────────────────────────────────────────────────────
int getArg(int argc, char** argv, const std::string& key, int def) {
  for (int i = 1; i < argc - 1; ++i)
    if (argv[i] == key) return std::stoi(argv[i + 1]);
  return def;
}
bool hasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (argv[i] == key) return true;
  return false;
}

// ── 随机 prompt ──────────────────────────────────────────────────────────────
VecTokens make_random_prompt(int length, unsigned seed = 42) {
  std::mt19937 rng(seed);
  // Qwen2.5 普通 token 范围，避开 151643+ 的特殊 token
  std::uniform_int_distribution<int> dist(100, 100000);
  VecTokens tokens(length);
  for (auto& t : tokens) t = dist(rng);
  return tokens;
}

// ── MPI sync（避免 LEADER 模式下 Barrier 死锁）────────────────────────────────
void sync(int rank) {
  if (rank == 0) {
    int s = 1;
    MPI_Send(&s, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&s, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int s = 1;
    MPI_Recv(&s, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&s, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

int main(int argc, char** argv) {
  // Usage:
  //   infer_bench <engine_dir>
  //     [--num_requests 100]
  //     [--concurrency  8]
  //     [--input_len    512]
  //     [--output_len   200]
  //     [--warmup       5]
  //     [--reuse_kv]     开启 prefix cache 复用（所有请求用同一个 prompt）
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_dir>"
                 " [--num_requests N]"
                 " [--concurrency C]"
                 " [--input_len I]"
                 " [--output_len O]"
                 " [--warmup W]"
                 " [--reuse_kv]\n";
    return 1;
  }

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  initTrtLlmPlugins();

  const int  num_requests = getArg(argc, argv, "--num_requests", 100);
  const int  concurrency  = getArg(argc, argv, "--concurrency",  8);
  const int  input_len    = getArg(argc, argv, "--input_len",    512);
  const int  output_len   = getArg(argc, argv, "--output_len",   200);
  const int  warmup       = getArg(argc, argv, "--warmup",       5);
  const bool reuse_kv     = hasFlag(argc, argv, "--reuse_kv");

  if (rank == 0)
    std::cout << "[Rank 0] Leader starting\n";
  else
    std::cout << "[Rank " << rank << "] Worker starting\n";

  auto parallel_config =
      ParallelConfig(CommunicationType::kMPI, CommunicationMode::kLEADER);

  KvCacheConfig kv_cache_config;
  kv_cache_config.setFreeGpuMemoryFraction(0.6f);
  kv_cache_config.setEnableBlockReuse(reuse_kv);

  ExecutorConfig executor_config(1);
  executor_config.setKvCacheConfig(kv_cache_config);
  executor_config.setParallelConfig(parallel_config);
  executor_config.setBatchingType(BatchingType::kINFLIGHT);

  if (rank == 0) std::cout << "[Rank 0] Loading engine...\n";
  auto executor = Executor(argv[1], ModelType::kDECODER_ONLY, executor_config);

  std::cout << "[Rank " << rank << "] Model loaded. Syncing...\n";
  sync(rank);

  // ── Worker rank：等待即可 ──────────────────────────────────────────────────
  if (rank != 0) {
    std::cout << "[Rank " << rank << "] Worker active, waiting...\n";
    sync(rank);
    executor.shutdown();
    MPI_Finalize();
    return 0;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Rank 0：压测主循环
  // ═══════════════════════════════════════════════════════════════════════════
  std::cout << "[Rank 0] Bench params:"
            << " num_requests=" << num_requests
            << " concurrency="  << concurrency
            << " input_len="    << input_len
            << " output_len="   << output_len
            << " warmup="       << warmup
            << " reuse_kv="     << (reuse_kv ? "on" : "off") << "\n";

  SamplingConfig sampling(1);

  // per-request 统计
  struct ReqStat {
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::high_resolution_clock::time_point t_first;
    bool first_seen  = false;
    int  total_tokens = 0;
  };
  std::map<IdType, ReqStat> stats;

  // 结果（warmup 之后的请求）
  std::vector<long> ttft_ms_vec, e2e_ms_vec;
  std::vector<int>  token_counts;
  ttft_ms_vec.reserve(num_requests);
  e2e_ms_vec.reserve(num_requests);
  token_counts.reserve(num_requests);

  int sent     = 0;
  int finished = 0;
  int total    = num_requests + warmup;

  // reuse_kv=on：所有请求用同一个 prompt（固定 seed），触发 prefix cache 命中
  // reuse_kv=off：每个请求不同 prompt（seed=sent），不命中
  auto send_one = [&]() {
    unsigned seed = reuse_kv ? 42 : (unsigned)sent;
    VecTokens prompt = make_random_prompt(input_len, seed);
    Request req(prompt, output_len, true, sampling);
    auto id = executor.enqueueRequest(req);
    stats[id].t_start = std::chrono::high_resolution_clock::now();
    ++sent;
  };

  // 初始填充
  int initial = std::min(concurrency, total);
  for (int i = 0; i < initial; ++i) send_one();

  auto bench_start = std::chrono::high_resolution_clock::now();

  // 主收集循环
  while (finished < total) {
    auto responses = executor.awaitResponses(std::chrono::milliseconds(50));
    for (auto& r : responses) {
      auto id  = r.getRequestId();
      auto& st = stats[id];

      if (r.hasError()) {
        std::cerr << "[Rank 0] Request " << id
                  << " error: " << r.getErrorMsg() << "\n";
        ++finished;
        if (sent < total) send_one();
        continue;
      }

      const auto& res = r.getResult();
      if (!res.outputTokenIds.empty()) {
        const auto& beam0 = res.outputTokenIds[0];
        if (!st.first_seen && !beam0.empty()) {
          st.t_first   = std::chrono::high_resolution_clock::now();
          st.first_seen = true;
        }
        st.total_tokens += (int)beam0.size();
      }

      if (res.isFinal) {
        auto t_end    = std::chrono::high_resolution_clock::now();
        bool is_warmup = (finished < warmup);

        if (!is_warmup && st.first_seen) {
          long ttft = std::chrono::duration_cast<std::chrono::milliseconds>(
                          st.t_first - st.t_start).count();
          long e2e  = std::chrono::duration_cast<std::chrono::milliseconds>(
                          t_end - st.t_start).count();
          ttft_ms_vec.push_back(ttft);
          e2e_ms_vec.push_back(e2e);
          token_counts.push_back(st.total_tokens);
        }

        ++finished;
        if (sent < total) send_one();
      }
    }
  }

  auto bench_end = std::chrono::high_resolution_clock::now();
  long wall_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                       bench_end - bench_start).count();

  // ── 统计 ────────────────────────────────────────────────────────────────────
  int N = (int)ttft_ms_vec.size();

  auto percentile = [](std::vector<long> v, double p) -> long {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    int idx = std::max(0, (int)(p / 100.0 * (int)v.size()) - 1);
    return v[idx];
  };

  long ttft_avg  = N ? std::accumulate(ttft_ms_vec.begin(), ttft_ms_vec.end(), 0LL) / N : 0;
  long e2e_avg   = N ? std::accumulate(e2e_ms_vec.begin(),  e2e_ms_vec.end(),  0LL) / N : 0;
  int  total_tok = std::accumulate(token_counts.begin(), token_counts.end(), 0);
  double throughput = wall_ms ? total_tok * 1000.0 / wall_ms : 0.0;

  // TPOT：每个请求 (e2e - ttft) / (tokens - 1) 取平均
  double tpot_sum = 0.0;
  int    tpot_cnt = 0;
  for (int i = 0; i < N; ++i) {
    if (token_counts[i] > 1) {
      tpot_sum += (double)(e2e_ms_vec[i] - ttft_ms_vec[i]) / (token_counts[i] - 1);
      ++tpot_cnt;
    }
  }
  double tpot_avg = tpot_cnt ? tpot_sum / tpot_cnt : 0.0;

  std::cout << "\n══════════════════ Bench Results ══════════════════\n";
  std::cout << "Requests       : " << N << " (warmup=" << warmup << " excluded)\n";
  std::cout << "Concurrency    : " << concurrency << "\n";
  std::cout << "Input / Output : " << input_len << " / " << output_len << " tokens\n";
  std::cout << "Wall time      : " << wall_ms << " ms\n";
  std::cout << "Total tokens   : " << total_tok << "\n";
  std::cout << "Throughput     : " << throughput << " tokens/s\n";
  std::cout << "──────────────────────────────────────────────────\n";
  std::cout << "TTFT avg       : " << ttft_avg << " ms\n";
  std::cout << "TTFT p50       : " << percentile(ttft_ms_vec, 50) << " ms\n";
  std::cout << "TTFT p99       : " << percentile(ttft_ms_vec, 99) << " ms\n";
  std::cout << "E2E  avg       : " << e2e_avg  << " ms\n";
  std::cout << "E2E  p50       : " << percentile(e2e_ms_vec,  50) << " ms\n";
  std::cout << "E2E  p99       : " << percentile(e2e_ms_vec,  99) << " ms\n";
  std::cout << "TPOT avg       : " << tpot_avg << " ms/token\n";
  std::cout << "══════════════════════════════════════════════════\n";

  sync(rank);
  executor.shutdown();
  MPI_Finalize();
  return 0;
}