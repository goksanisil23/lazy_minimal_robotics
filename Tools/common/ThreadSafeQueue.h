#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template <class T>
class TSQueue
{
  public:
    TSQueue() : queue_(), m_(), cond_(), queue_size_limit_{20}
    {
    }

    // ~TSQueue() {};

    // Add element to the queue
    void Enqueue(const T &el)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (queue_.size() > queue_size_limit_) // remove oldest element if the buffer grew too much
        {
            queue_.pop();
        }
        queue_.push(el);
        cond_.notify_one();
    }

    // Get the front element, if queue is empty, wait till available
    // by releasing the lock temporarily and reacquiring after non-empty
    void Dequeue(T &el)
    {
        std::unique_lock<std::mutex> lock(m_);
        while (queue_.empty())
        {
            // lock is release while waiting, will be waken up with notify_one or notify_all
            cond_.wait(lock);
        }
        el = queue_.front();
        queue_.pop();
    }

    size_t GetSize() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return queue_.size();
    }

  private:
    std::queue<T>           queue_;
    mutable std::mutex      m_;
    std::condition_variable cond_;
    const uint8_t           queue_size_limit_; // to avoid the buffer growing too big
};