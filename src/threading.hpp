#pragma once

#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <vector>
#include <iostream>

#include "queue.hpp"

/**
    * @class ThreadPool
    * @brief A thread pool implementation for executing tasks concurrently.
    *
    * ThreadPool manages a collection of threads that process tasks from a shared queue.
    * It provides functionality to add tasks, wait for completion, and properly shut down the pool.
*/
class ThreadPool{
    private:
        /**
            * @brief Flag indicating whether the thread pool should shut down.
            * 
            * When set to true, worker threads will exit after completing their current tasks.
        */
        bool exit = false;
    
        /**
            * @brief Vector of worker threads managed by the pool.
        */
        std::vector<std::thread> threads;
        
        /**
            * @brief Queue containing pending tasks to be executed.
        */
        Queue<std::function<void()>> taskQueue;

        /**
            * @brief Mutex for synchronizing access to the thread pool's state.
        */
        mutable std::mutex poolTex;
        
        /**
            * @brief Condition variable for signaling when work is available.
        */
        mutable std::condition_variable cond_work;

        /**
            * @brief Counter tracking the number of busy threads.
        */
        unsigned int busy = 0;
        
        /**
            * @brief Condition variable for signaling when all tasks are completed.
        */
        mutable std::condition_variable cond_finished;

    public:
        /**
            * @brief Constructs a ThreadPool with the specified number of worker threads.
            * 
            * @param num_threads The number of threads to create in the pool. Defaults to the number of hardware threads available on the system.
        */
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()){
            for (size_t i = 0; i < num_threads; ++i) {
                threads.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        // open a scope for unique_lock
                        {
                            // lock queue
                            std::unique_lock<std::mutex> lock(poolTex);
    
                            // wait for task to be put in queue
                            cond_work.wait(lock, [this] {
                                return !taskQueue.isEmpty() || exit;
                            });
    
                            // pool is stopped and there are no more tasks
                            if (exit && taskQueue.isEmpty()) {
                                return;
                            }
                            
                            task = std::move(taskQueue.deQueue());
                            busy++;

                            // Unlock to execute task
                            lock.unlock();

                            task();

                            // Lock to notify end of job
                            lock.lock();
                            busy--;
                            cond_finished.notify_one();
                        }
                    }
                });
            }
        };

        /**
            * @brief Destroys the ThreadPool, stopping all worker threads.
            * 
            * Sets the exit flag and waits for all threads to complete their tasks and terminate.
        */
        ~ThreadPool(){
            {
                std::unique_lock<std::mutex> lock(poolTex);
                exit = true;
            }
        
            cond_work.notify_all();
        
            for (auto& thread : threads) {
                thread.join();
            }
        }

        /**
            * @brief Adds a new task to the thread pool's queue.
            * 
            * @param task A function that will be executed by one of the worker threads.
        */
        void addTask(std::function<void()>&& task){
            {
                std::unique_lock<std::mutex> lock(poolTex);
                taskQueue.enQueue(std::move(task));
            }
            cond_work.notify_one();
        }

        /**
            * @brief Blocks until all tasks in the queue are completed.
            * 
            * This method waits until the task queue is empty and all worker threads are idle.
        */
        void waitFinished(){
            std::unique_lock<std::mutex> lock(poolTex);
            cond_finished.wait(lock, [this](){ return taskQueue.isEmpty() && (busy == 0); });
        }
};