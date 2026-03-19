#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>

class MpiBufferManager {
public:
    // RAII 风格：构造时 Attach，析构时自动 Detach 并释放内存
    MpiBufferManager(int data_size, MPI_Datatype type, MPI_Comm comm) {
        int bsize;
        MPI_Pack_size(data_size, type, comm, &bsize);
        
        // 申请内存：数据大小 + MPI 开销
        buffer_storage_.resize(bsize + MPI_BSEND_OVERHEAD);
        MPI_Buffer_attach(buffer_storage_.data(), buffer_storage_.size());
        std::cerr << "[Log] MPI Buffer attached (" << buffer_storage_.size() << " bytes)\n";
    }

    ~MpiBufferManager() {
        void* ptr;
        int size;
        MPI_Buffer_detach(&ptr, &size);
        std::cerr << "[Log] MPI Buffer detached and cleaned up\n";
    }

private:
    std::vector<char> buffer_storage_;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "Error: This demo requires 2 processes.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int DATA_SIZE = 6;
    const int TAG = 2000;

    if (rank == 0) { // 发送端
        std::vector<double> data(DATA_SIZE);
        std::iota(data.begin(), data.end(), 1.0); // 填充 1, 2, 3...

        {
            // 利用作用域管理缓冲区周期
            MpiBufferManager manager(DATA_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

            std::cerr << "Rank 0: Bsending " << DATA_SIZE - 1 << " elements...\n";
            MPI_Bsend(data.data(), DATA_SIZE - 1, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);

            std::cerr << "Rank 0: Bsending last 1 element...\n";
            MPI_Bsend(&data[DATA_SIZE - 1], 1, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);
        } // 此处 manager 析构，自动执行 Detach

    } else { // 接收端
        std::vector<double> recv_buf(DATA_SIZE);
        MPI_Status status;

        MPI_Recv(recv_buf.data(), DATA_SIZE - 1, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&recv_buf[DATA_SIZE - 1], 1, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);

        std::cout << "Rank 1 received: ";
        for (auto val : recv_buf) std::cout << val << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}