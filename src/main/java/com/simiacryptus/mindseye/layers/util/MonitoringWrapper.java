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

package com.simiacryptus.mindseye.layers.util;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.ScalarStatistics;
import com.simiacryptus.util.ml.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("serial")
public final class MonitoringWrapper extends NNLayerWrapper implements MonitoredItem {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    //json.add("forwardPerf",forwardPerf.getJson());
    //json.add("backwardPerf",backwardPerf.getJson());
    json.add("inner",inner.getJson());
    json.addProperty("totalBatches",totalBatches);
    json.addProperty("totalItems",totalItems);
    return json;
  }
  public static MonitoringWrapper fromJson(JsonObject json) {
    return new MonitoringWrapper(json);
  }
  protected MonitoringWrapper(JsonObject json) {
    super(json);
    if(json.has("forwardPerf")) this.forwardPerf.readJson(json.getAsJsonObject("forwardPerf"));
    if(json.has("backwardPerf")) this.backwardPerf.readJson(json.getAsJsonObject("backwardPerf"));
    this.totalBatches = json.get("totalBatches").getAsInt();
    this.totalItems = json.get("totalItems").getAsInt();
  }
  
  private final ScalarStatistics forwardPerf = new ScalarStatistics();
  private final ScalarStatistics backwardPerf = new ScalarStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;
  
  public MonitoringWrapper(final NNLayer inner) {
    super(inner);
  }
  
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forwardPerformance", forwardPerf.getMetrics());
    map.put("backwardPerformance", backwardPerf.getMetrics());
    double batchesPerItem = totalBatches * 1.0 / totalItems;
    map.put("avgMsPerItem", 1000 * batchesPerItem * forwardPerf.getMean());
    map.put("avgMsPerItem_Backward", 1000 * batchesPerItem * backwardPerf.getMean());
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
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    long start = System.nanoTime();
    final NNResult result = this.inner.eval(inObj);
    forwardPerf.add(((System.nanoTime() - start) / 1000000000.0));
    totalBatches++;
    totalItems += inObj[0].data.length;
    return new NNResult(result.data) {
      @Override
      public void accumulate(DeltaSet buffer, Tensor[] data) {
        long start = System.nanoTime();
        result.accumulate(buffer, data);
        backwardPerf.add(((System.nanoTime() - start) / 1000000000.0));
      }
  
      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    };
  }
  
  public MonitoringWrapper addTo(MonitoredObject obj) {
    return addTo(obj, inner.getName());
  }
  
  public MonitoringWrapper addTo(MonitoredObject obj, String name) {
    setName(name);
    obj.addObj(getName(),this);
    return this;
  }
  
}
