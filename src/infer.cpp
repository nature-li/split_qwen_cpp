#include <mpi.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <chrono>
#include <iostream>

using namespace tensorrt_llm::executor;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <engine_dir>\n";
    return 1;
  }

  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  initTrtLlmPlugins();

  // ExecutorConfig executor_config(1);
  KvCacheConfig kv_cache_config;
  kv_cache_config.setFreeGpuMemoryFraction(0.5f);  // 只用50%显存
  ExecutorConfig executor_config(1, SchedulerConfig(), kv_cache_config);

  auto executor = Executor(argv[1], ModelType::kDECODER_ONLY, executor_config);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    // enqueue 请求
    VecTokens input_ids = {151644, 8948,  198,    2610,   525,   264,
                           10950,  17847, 13,     151645, 198,   151644,
                           872,    198,   109432, 102064, 15471, 5327,
                           151645, 198,   151644, 77091,  198};
    SamplingConfig sampling(1);
    const int32_t max_new_tokens = 200;
    Request req(input_ids, max_new_tokens, true, sampling);
    auto req_id = executor.enqueueRequest(req);
    std::cout << "Request enqueued\n" << std::flush;
  }

  // 两边都跑 awaitResponses 循环
  VecTokens output_tokens;
  bool done = false;
  auto t0 = std::chrono::high_resolution_clock::now();

  while (!done) {
    auto responses = executor.awaitResponses(std::chrono::milliseconds(200));
    for (const auto& r : responses) {
      if (r.hasError()) {
        std::cerr << "Error: " << r.getErrorMsg() << "\n";
        executor.shutdown();
        MPI_Finalize();
        return 1;
      }
      const auto& res = r.getResult();
      if (!res.outputTokenIds.empty()) {
        const auto& beam0 = res.outputTokenIds[0];
        output_tokens.insert(output_tokens.end(), beam0.begin(), beam0.end());
      }
      if (res.isFinal) done = true;
    }
  }

  if (rank == 0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Generated " << output_tokens.size() << " tokens in " << ms
              << " ms\n";
    std::cout << "Throughput: " << output_tokens.size() * 1000.0 / ms
              << " tokens/s\n";
    std::cout << "Token ids: ";
    for (auto t : output_tokens) std::cout << t << " ";
    std::cout << "\n";
  }

  executor.shutdown();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}