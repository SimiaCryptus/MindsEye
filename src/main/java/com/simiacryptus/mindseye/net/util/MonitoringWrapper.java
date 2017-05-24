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
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.ScalarStatistics;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("serial")
public final class MonitoringWrapper extends NNLayer implements MonitoredItem {
  
  public final NNLayer inner;
  private double totalTime = 0;
  private int totalBatches = 0;
  private double totalTimeSq = 0;
  private int totalItems = 0;
  private boolean enabled = false;
  
  public MonitoringWrapper(final NNLayer inner) {
    this.inner = inner;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    long start = System.nanoTime();
    final NNResult result = this.inner.eval(inObj);
    double elapsed = (System.nanoTime() - start) / 1000000000.0;
    totalTime += elapsed;
    totalTimeSq += elapsed*elapsed;
    totalBatches++;
    totalItems += inObj[0].data.length;
    return result;
  }
  
  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    double mean = totalTime / totalBatches;
    map.put("avgMsPerBatch", 1000*mean);
    map.put("avgMsPerItem", 1000*totalTime / totalItems);
    map.put("stddevMsPerBatch", 1000*Math.sqrt(Math.abs(totalTimeSq/totalItems - mean*mean)));
    List<double[]> state = state();
    HashMap<String, Object> weightStats = new HashMap<>();
    map.put("weights", weightStats);
    weightStats.put("buffers", state.size());
    ScalarStatistics statistics = new ScalarStatistics();
    for(double[] s : state) {
      for(double v : s) {
        statistics.add(v);
      }
    }
    weightStats.putAll(statistics.getMetrics());
    return map;
  }
  
  public MonitoringWrapper addTo(MonitoredObject obj, String name) {
    this.enabled = true;
    obj.addObj(name,this);
    return this;
  }
}
