#include <mpi.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <chrono>
#include <iostream>

using namespace tensorrt_llm::executor;

void sync(int rank) {
  if (rank == 0) {
    int signal = 1;
    MPI_Send(&signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    int signal = 1;
    MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <engine_dir>\n";
    return 1;
  }

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  initTrtLlmPlugins();

  if (rank == 0)
    std::cout << "[Rank 0] Leader starting on naked\n";
  else
    std::cout << "[Rank " << rank << "] Worker starting\n";

  auto parallel_config =
      ParallelConfig(CommunicationType::kMPI, CommunicationMode::kLEADER);

  KvCacheConfig kv_cache_config;
  kv_cache_config.setFreeGpuMemoryFraction(0.4f);

  ExecutorConfig executor_config(1);
  executor_config.setKvCacheConfig(kv_cache_config);
  executor_config.setParallelConfig(parallel_config);
  executor_config.setBatchingType(BatchingType::kINFLIGHT);

  if (rank == 0) std::cout << "[Rank 0] Loading engine...\n";
  auto executor = Executor(argv[1], ModelType::kDECODER_ONLY, executor_config);

  std::cout << "[Rank " << rank << "] Model loaded. Syncing...\n";
  sync(rank);  // 替代第一个 MPI_Barrier

  if (rank == 0) {
    std::cout << "[Rank 0] All synced. Sending request...\n";

    VecTokens input_ids = {151644, 8948,   198,   2610,  525,    1207,   16948,
                           11,     3465,   553,   54364, 14817,  13,     1446,
                           525,    264,    10950, 17847, 13,     151645, 198,
                           151644, 872,    198,   99526, 100158, 42578,  151645,
                           198,    151644, 77091, 198};
    SamplingConfig sampling(1);
    Request req(input_ids, 200, true, sampling);

    auto req_id = executor.enqueueRequest(req);
    std::cout << "[Rank 0] Request " << req_id << " enqueued.\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t_first;
    bool first_token = false;

    VecTokens output_tokens;
    bool done = false;

    while (!done) {
      auto responses = executor.awaitResponses(std::chrono::milliseconds(100));
      for (const auto& r : responses) {
        if (r.hasError()) {
          std::cerr << "[Rank 0] Error: " << r.getErrorMsg() << "\n";
          done = true;
          break;
        }
        const auto& res = r.getResult();
        if (!res.outputTokenIds.empty()) {
          const auto& beam0 = res.outputTokenIds[0];
          if (!first_token && !beam0.empty()) {
            t_first = std::chrono::high_resolution_clock::now();
            first_token = true;
          }
          output_tokens.insert(output_tokens.end(), beam0.begin(), beam0.end());
        }
        if (res.isFinal) done = true;
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ttft_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_first - t0)
            .count();
    int n = output_tokens.size();
    double tpot = n > 1 ? (double)(total_ms - ttft_ms) / (n - 1) : 0.0;

    std::cout << "\n=== Performance ===\n";
    std::cout << "Total tokens : " << n << "\n";
    std::cout << "Total time   : " << total_ms << " ms\n";
    std::cout << "TTFT         : " << ttft_ms << " ms\n";
    std::cout << "TPOT         : " << tpot << " ms/token\n";
    std::cout << "Throughput   : " << n * 1000.0 / (total_ms ? total_ms : 1)
              << " tokens/s\n";

    std::cout << "\nToken IDs: ";
    for (auto t : output_tokens) std::cout << t << " ";
    std::cout << "\n";

  } else {
    std::cout << "[Rank " << rank << "] Worker active, waiting...\n";
  }

  sync(rank);  // 替代第二个 MPI_Barrier
  executor.shutdown();
  MPI_Finalize();
  return 0;
}