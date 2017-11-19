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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The type Monitoring wrapper.
 */
@SuppressWarnings("serial")
public final class MonitoringWrapper extends NNLayerWrapper implements MonitoredItem {
  
  private boolean verbose = false;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    //json.fn("forwardPerf",forwardPerf.getJson());
    //json.fn("backwardPerf",backwardPerf.getJson());
    json.add("inner", getInner().getJson());
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    json.addProperty("recordSignalMetrics", recordSignalMetrics);
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
    if (json.has("forwardPerf")) this.forwardPerformance.readJson(json.getAsJsonObject("forwardPerf"));
    if (json.has("backwardPerf")) this.backwardPerformance.readJson(json.getAsJsonObject("backwardPerf"));
    if (json.has("backpropStatistics")) this.backwardSignal.readJson(json.getAsJsonObject("backpropStatistics"));
    if (json.has("outputStatistics")) this.forwardSignal.readJson(json.getAsJsonObject("outputStatistics"));
    this.recordSignalMetrics = json.get("recordSignalMetrics").getAsBoolean();
    this.totalBatches = json.get("totalBatches").getAsInt();
    this.totalItems = json.get("totalItems").getAsInt();
  }
  
  private final PercentileStatistics forwardPerformance = new PercentileStatistics();
  private final PercentileStatistics backwardPerformance = new PercentileStatistics();
  private final ScalarStatistics backwardSignal = new PercentileStatistics();
  private final ScalarStatistics forwardSignal = new PercentileStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;
  private boolean recordSignalMetrics = true;
  
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
    map.put("outputStatistics", forwardSignal.getMetrics());
    map.put("backpropStatistics", backwardSignal.getMetrics());
    if (verbose) {
      map.put("forwardPerformance", forwardPerformance.getMetrics());
      map.put("backwardPerformance", backwardPerformance.getMetrics());
    }
    double batchesPerItem = totalBatches * 1.0 / totalItems;
    map.put("avgMsPerItem", 1000 * batchesPerItem * forwardPerformance.getMean());
    map.put("medianMsPerItem", 1000 * batchesPerItem * forwardPerformance.getPercentile(0.5));
    double backpropMean = backwardPerformance.getMean();
    double backpropMedian = backwardPerformance.getPercentile(0.5);
    map.put("avgMsPerItem_Backward", 1000 * batchesPerItem * backpropMean);
    map.put("medianMsPerItem_Backward", 1000 * batchesPerItem * backpropMedian);
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
    AtomicLong passback = new AtomicLong(0);
    NNResult[] wrappedInput = Arrays.stream(inObj).map(result -> new NNResult(result.getData()) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        long start = System.nanoTime();
        result.accumulate(buffer, data);
        long elapsed = System.nanoTime() - start;
        passback.addAndGet(elapsed);
      }
      
      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    }).toArray(i -> new NNResult[i]);
    long start = System.nanoTime();
    final NNResult output = this.getInner().eval(nncontext, wrappedInput);
    forwardPerformance.add(((System.nanoTime() - start) / 1000000000.0));
    totalBatches++;
    int items = Arrays.stream(inObj).mapToInt(x -> x.getData().length()).max().orElse(1);
    totalItems +=  items;
    if(recordSignalMetrics) {
      forwardSignal.clear();
      output.getData().stream().parallel().forEach(t -> {
        forwardSignal.add(t.getData());
      });
    }
    return new NNResult(output.getData()) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        if(recordSignalMetrics) {
          backwardSignal.clear();
          data.stream().parallel().forEach(t -> {
            backwardSignal.add(t.getData());
          });
        }
        long start = System.nanoTime();
        output.accumulate(buffer, data);
        backwardPerformance.add(((System.nanoTime() - start)  - passback.getAndSet(0)) / (items*1e9));
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
  
  /**
   * Is activity stats boolean.
   *
   * @return the boolean
   */
  public boolean recordSignalMetrics() {
    return recordSignalMetrics;
  }
  
  /**
   * Sets activity stats.
   *
   * @param recordSignalMetrics the activity stats
   * @return the activity stats
   */
  public MonitoringWrapper shouldRecordSignalMetrics(boolean recordSignalMetrics) {
    this.recordSignalMetrics = recordSignalMetrics;
    return this;
  }
  
  public PercentileStatistics getForwardPerformance() {
    return forwardPerformance;
  }
  
  public PercentileStatistics getBackwardPerformance() {
    return backwardPerformance;
  }
  
  public ScalarStatistics getBackwardSignal() {
    return backwardSignal;
  }
  
  public ScalarStatistics getForwardSignal() {
    return forwardSignal;
  }
}
