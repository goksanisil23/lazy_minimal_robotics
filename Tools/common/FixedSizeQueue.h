#pragma once

#include <deque>

template <typename T>
class FixedSizeQueue
{
  public:
    FixedSizeQueue() = default;

    FixedSizeQueue(const size_t &size) : size_(size)
    {
    }

    template <typename TT>
    void push(TT &&input)
    {
        if (dq_.size() == size_)
        {
            dq_.pop_front();
        }
        dq_.push_back(std::forward<TT>(input));
    }

    auto front()
    {
        return dq_.front();
    }

    auto back()
    {
        return dq_.back();
    }

    size_t size() const
    {
        return size();
    }

    bool empty() const
    {
        return dq_.empty();
    }

    auto begin() const
    {
        return dq_.begin();
    }

    auto end() const
    {
        return dq_.end();
    }

    auto at(const size_t &idx)
    {
        return dq_.at(idx);
    }

  private:
    size_t        size_;
    std::deque<T> dq_;
};