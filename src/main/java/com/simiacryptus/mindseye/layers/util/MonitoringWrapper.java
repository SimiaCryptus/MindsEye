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
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.PercentileStatistics;
import com.simiacryptus.util.ScalarStatistics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The type Monitoring wrapper.
 */
@SuppressWarnings("serial")
public final class MonitoringWrapper extends NNLayerWrapper implements MonitoredItem {
  
  private boolean verbose = false;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    //json.add("forwardPerf",forwardPerf.getJson());
    //json.add("backwardPerf",backwardPerf.getJson());
    json.add("inner", getInner().getJson());
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }
  
  /**
   * From json monitoring wrapper.
   *
   * @param json the json
   * @return the monitoring wrapper
   */
  public static MonitoringWrapper fromJson(JsonObject json) {
    return new MonitoringWrapper(json);
  }
  
  /**
   * Instantiates a new Monitoring wrapper.
   *
   * @param json the json
   */
  protected MonitoringWrapper(JsonObject json) {
    super(json);
    if (json.has("forwardPerf")) this.forwardPerf.readJson(json.getAsJsonObject("forwardPerf"));
    if (json.has("backwardPerf")) this.backwardPerf.readJson(json.getAsJsonObject("backwardPerf"));
    if (json.has("passbackPerformance")) this.passbackPerf.readJson(json.getAsJsonObject("passbackPerformance"));
    if (json.has("backpropStatistics")) this.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    if (json.has("outputStatistics")) this.outputStatistics.readJson(json.getAsJsonObject("outputStatistics"));
    this.totalBatches = json.get("totalBatches").getAsInt();
    this.totalItems = json.get("totalItems").getAsInt();
  }
  
  private final PercentileStatistics passbackPerf = new PercentileStatistics();
  private final PercentileStatistics forwardPerf = new PercentileStatistics();
  private final PercentileStatistics backwardPerf = new PercentileStatistics();
  private final ScalarStatistics backpropStatistics = new PercentileStatistics();
  private final ScalarStatistics outputStatistics = new PercentileStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;
  private boolean activityStats = true;
  
  /**
   * Instantiates a new Monitoring wrapper.
   *
   * @param inner the inner
   */
  public MonitoringWrapper(final NNLayer inner) {
    super(inner);
  }
  
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("class", inner.getClass().getName());
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("outputStatistics", outputStatistics.getMetrics());
    map.put("backpropStatistics", backpropStatistics.getMetrics());
    if (verbose) {
      map.put("forwardPerformance", forwardPerf.getMetrics());
      map.put("backwardPerformance", backwardPerf.getMetrics());
      map.put("passbackPerformance", passbackPerf.getMetrics());
    }
    double batchesPerItem = totalBatches * 1.0 / totalItems;
    map.put("avgMsPerItem", 1000 * batchesPerItem * forwardPerf.getMean());
    map.put("medianMsPerItem", 1000 * batchesPerItem * forwardPerf.getPercentile(0.5));
    double passbackMean = passbackPerf.getMean();
    double backpropMean = backwardPerf.getMean();
    double passbackMedian = passbackPerf.getPercentile(0.5);
    double backpropMedian = backwardPerf.getPercentile(0.5);
    map.put("avgMsPerItem_Backward", 1000 * batchesPerItem * (Double.isFinite(passbackMean) ? (backpropMean - passbackMean) : backpropMean));
    map.put("medianMsPerItem_Backward", 1000 * batchesPerItem * (Double.isFinite(passbackMedian) ? (backpropMedian - passbackMedian) : backpropMedian));
    List<double[]> state = state();
    ScalarStatistics statistics = new PercentileStatistics();
    for (double[] s : state) {
      for (double v : s) {
        statistics.add(v);
      }
    }
    if (statistics.getCount() > 0) {
      HashMap<String, Object> weightStats = new HashMap<>();
      weightStats.put("buffers", state.size());
      weightStats.putAll(statistics.getMetrics());
      map.put("weights", weightStats);
    }
    return map;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    long passbacks = Arrays.stream(inObj).filter(x -> x.isAlive()).count();
    NNResult[] wrappedInput = Arrays.stream(inObj).map(result -> new NNResult(result.getData()) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        long start = System.nanoTime();
        result.accumulate(buffer, data);
        passbackPerf.add((passbacks * (System.nanoTime() - start) / 1000000000.0));
      }
      
      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    }).toArray(i -> new NNResult[i]);
    long start = System.nanoTime();
    final NNResult output = this.getInner().eval(nncontext, wrappedInput);
    forwardPerf.add(((System.nanoTime() - start) / 1000000000.0));
    totalBatches++;
    totalItems += inObj[0].getData().length();
    if(isActivityStats()) {
      outputStatistics.clear();
      output.getData().stream().parallel().forEach(t -> {
        outputStatistics.add(t.getData());
      });
    }
    return new NNResult(output.getData()) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        if(isActivityStats()) {
          backpropStatistics.clear();
          data.stream().parallel().forEach(t -> {
            backpropStatistics.add(t.getData());
          });
        }
        long start = System.nanoTime();
        output.accumulate(buffer, data);
        backwardPerf.add(((System.nanoTime() - start) / 1000000000.0));
      }
      
      @Override
      public boolean isAlive() {
        return output.isAlive();
      }
    };
  }
  
  /**
   * Add to monitoring wrapper.
   *
   * @param obj the obj
   * @return the monitoring wrapper
   */
  public MonitoringWrapper addTo(MonitoredObject obj) {
    return addTo(obj, getInner().getName());
  }
  
  /**
   * Add to monitoring wrapper.
   *
   * @param obj  the obj
   * @param name the name
   * @return the monitoring wrapper
   */
  public MonitoringWrapper addTo(MonitoredObject obj, String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }
  
  @Override
  public String getName() {
    return getInner().getName();
  }
  
  @Override
  public NNLayer setName(String name) {
    if (null != getInner()) getInner().setName(name);
    return this;
  }
  
  public boolean isActivityStats() {
    return activityStats;
  }
  
  public MonitoringWrapper setActivityStats(boolean activityStats) {
    this.activityStats = activityStats;
    return this;
  }
}
