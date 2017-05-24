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

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.ScalarStatistics;
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
    NNResult input = inObj[0];
    if(enabled) {
      long start = System.nanoTime();
      double elapsed = (System.nanoTime() - start) / 1000000000.0;
      totalBatches++;
      totalItems += input.data.length;
      forwardStatistics.clear();
      for(Tensor t : input.data) {
        for(double v : t.getData()) {
          forwardStatistics.add(v);
        }
      }
    }
    return new NNResult(input.data) {
      @Override
      public void accumulate(DeltaSet buffer, Tensor[] data) {
        backpropStatistics.clear();
        for(Tensor t : input.data) {
          for(double v : t.getData()) {
            backpropStatistics.add(v);
          }
        }
        input.accumulate(buffer, data);
      }
  
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  private final ScalarStatistics backpropStatistics = new ScalarStatistics();
  private final ScalarStatistics forwardStatistics = new ScalarStatistics();
  boolean enabled = false;
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forward", forwardStatistics.getMetrics());
    map.put("backprop", backpropStatistics.getMetrics());
    return map;
  }
  
  public MonitoringSynapse addTo(MonitoredObject obj, String name) {
    this.enabled = true;
    obj.addObj(name,this);
    return this;
  }
}
