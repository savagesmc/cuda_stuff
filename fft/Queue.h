#ifndef UTIL_QUEUE_H
#define UTIL_QUEUE_H
#pragma once

#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <boost/optional.hpp>

namespace Util
{

template<class T>
class Queue
{
   std::mutex mtx_;
   std::condition_variable cv_;
   std::deque<T> q_;
   const std::chrono::milliseconds timeout_;
public:
   Queue()
   : mtx_()
     , cv_()
     , q_()
     , timeout_(std::chrono::milliseconds(250))
   { }

   void push(T val)
   {
      {
         std::unique_lock<std::mutex> lock(mtx_);
         q_.push_back(std::move(val));
      }
      cv_.notify_one();
   }

   boost::optional<T> pop()
   {
      T result;
      std::chrono::system_clock::time_point expire
            = std::chrono::system_clock::now() + timeout_;
      std::unique_lock<std::mutex> lock(mtx_);
      while (q_.empty())
      {
         if (cv_.wait_until(lock, expire) == std::cv_status::timeout)
         {
            return boost::none;
         }
      }
      result = std::move(q_.front());
      q_.pop_front();
      return result;
   }
};

}
#endif
