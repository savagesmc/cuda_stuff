#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

class TimeStat
{
   std::chrono::steady_clock::time_point start;
   std::string label;
   int numIter_;
public:
   TimeStat(const std::string& s, int numIter=1)
   : label(s), start(std::chrono::steady_clock::now()), numIter_(numIter)
   { }
   ~TimeStat() {
      using namespace std;
      using namespace std::chrono;
      if (numIter_)
      {
         steady_clock::time_point stop = steady_clock::now();
         duration<double> dur = (stop - start) / numIter_;
         cout << label << " " << (dur.count()) << endl;
      }
   }
};
