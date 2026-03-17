#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

// 使用枚舉類管理標籤，避免魔法數字 (Magic Numbers)
enum class MsgTag {
    Exit = 1,
    Ordered = 2,
    Unordered = 3
};

class Node {
public:
    Node() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    void run() {
        if (rank_ == 0) master_job();
        else           slave_job();
    }

private:
    int rank_, size_;
    static constexpr int BUF_SIZE = 256;

    // --- 主進程邏輯 ---
    void master_job() {
        int active_slaves = size_ - 1;
        std::vector<char> buffer(BUF_SIZE);
        MPI_Status status;

        while (active_slaves > 0) {
            // 接收來自任何人的任何消息
            MPI_Recv(buffer.data(), BUF_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            MsgTag tag = static_cast<MsgTag>(status.MPI_TAG);
            int sender = status.MPI_SOURCE;

            switch (tag) {
                case MsgTag::Exit:
                    active_slaves--;
                    break;

                case MsgTag::Unordered:
                    std::cout << "[Unordered] " << buffer.data() << std::flush;
                    break;

                case MsgTag::Ordered: {
                    // 進入「強制排隊模式」：按順序點名接收 1 到 size-1 的消息
                    for (int i = 1; i < size_; ++i) {
                        if (i == sender) {
                            std::cout << "[Ordered] " << buffer.data() << std::flush;
                        } else {
                            std::vector<char> sync_buf(BUF_SIZE);
                            MPI_Recv(sync_buf.data(), BUF_SIZE, MPI_CHAR, i, static_cast<int>(MsgTag::Ordered), MPI_COMM_WORLD, &status);
                            std::cout << "[Ordered] " << sync_buf.data() << std::flush;
                        }
                    }
                    break;
                }
            }
        }
        std::cout << "Master: All slaves exited. System shutdown." << std::endl;
    }

    // --- 從進程邏輯 ---
    void slave_job() {
        auto send_msg = [this](const std::string& text, MsgTag tag) {
            MPI_Send(text.c_str(), text.size() + 1, MPI_CHAR, 0, static_cast<int>(tag), MPI_COMM_WORLD);
        };

        // 1. 發送有序消息
        send_msg("Hello from slave " + std::to_string(rank_) + " (Part A)", MsgTag::Ordered);
        send_msg("Goodbye from slave " + std::to_string(rank_) + " (Part B)", MsgTag::Ordered);

        // 2. 發送無序消息
        send_msg("I'm feeling chaotic (" + std::to_string(rank_) + ")", MsgTag::Unordered);

        // 3. 退出
        MPI_Send(nullptr, 0, MPI_CHAR, 0, static_cast<int>(MsgTag::Exit), MPI_COMM_WORLD);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    Node node;
    node.run();

    MPI_Finalize();
    return 0;
}