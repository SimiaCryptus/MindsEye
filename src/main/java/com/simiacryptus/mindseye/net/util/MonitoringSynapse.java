/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.net.util;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("serial")
public final class MonitoringSynapse extends NNLayer implements MonitoredItem {
  
  private int totalBatches = 0;
  private int totalItems = 0;
  
  public MonitoringSynapse() {
  
  
  }
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert(1==inObj.length);
    if(enabled) {
      long start = System.nanoTime();
      double elapsed = (System.nanoTime() - start) / 1000000000.0;
      totalBatches++;
      totalItems += inObj[0].data.length;
      zeros = 0;
      sum0 = 0;
      sum1 = 0;
      sum2 = 0;
      sumLog = 0;
      for(Tensor t : inObj[0].data) {
        for(double v : t.getData()) {
          sum0 += 1;
          sum1 += v;
          sum2 += v * v;
          if(Math.abs(v) < 1e-20) {
            zeros++;
          } else {
            sumLog += Math.log(Math.abs(v)) / Math.log(10);
          }
        }
      }
    }
    return inObj[0];
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  int zeros = 0;
  double sum0 = 0;
  double sum1 = 0;
  double sum2 = 0;
  double sumLog = 0;
  boolean enabled = false;
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("count", sum0);
    map.put("mean", sum1 / sum0);
    map.put("stdDev", Math.sqrt(Math.abs(Math.pow(sum1 / sum0, 2) - sum2 / sum0)));
    map.put("meanExponent", sumLog / (sum0-zeros));
    map.put("zeros", zeros);
    return map;
  }
  
  public MonitoringSynapse addTo(MonitoredObject obj, String name) {
    this.enabled = true;
    obj.addObj(name,this);
    return this;
  }
}
