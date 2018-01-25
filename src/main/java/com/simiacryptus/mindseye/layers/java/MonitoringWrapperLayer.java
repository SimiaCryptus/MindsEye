/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.util.lang.TimedResult;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A diagnostic wrapper that collects performance data and, if enabled, value statistics of output and backprop
 * signals.
 */
@SuppressWarnings("serial")
public final class MonitoringWrapperLayer extends WrapperLayer implements MonitoredItem {
  
  private final PercentileStatistics backwardPerformance = new PercentileStatistics();
  private final ScalarStatistics backwardSignal = new PercentileStatistics();
  private final PercentileStatistics forwardPerformance = new PercentileStatistics();
  private final ScalarStatistics forwardSignal = new PercentileStatistics();
  private final boolean verbose = false;
  private boolean recordSignalMetrics = false;
  private int totalBatches = 0;
  private int totalItems = 0;
  
  /**
   * Instantiates a new Monitoring wrapper layer.
   *
   * @param json the json
   */
  protected MonitoringWrapperLayer(final JsonObject json) {
    super(json);
    if (json.has("forwardPerf")) {
      forwardPerformance.readJson(json.getAsJsonObject("forwardPerf"));
    }
    if (json.has("backwardPerf")) {
      backwardPerformance.readJson(json.getAsJsonObject("backwardPerf"));
    }
    if (json.has("backpropStatistics")) {
      backwardSignal.readJson(json.getAsJsonObject("backpropStatistics"));
    }
    if (json.has("outputStatistics")) {
      forwardSignal.readJson(json.getAsJsonObject("outputStatistics"));
    }
    recordSignalMetrics = json.get("recordSignalMetrics").getAsBoolean();
    totalBatches = json.get("totalBatches").getAsInt();
    totalItems = json.get("totalItems").getAsInt();
  }
  
  /**
   * Instantiates a new Monitoring wrapper layer.
   *
   * @param inner the heapCopy
   */
  public MonitoringWrapperLayer(final NNLayer inner) {
    super(inner);
  }
  
  /**
   * From json monitoring wrapper layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the monitoring wrapper layer
   */
  public static MonitoringWrapperLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new MonitoringWrapperLayer(json);
  }
  
  /**
   * Add to monitoring wrapper layer.
   *
   * @param obj the obj
   * @return the monitoring wrapper layer
   */
  public MonitoringWrapperLayer addTo(final MonitoredObject obj) {
    return addTo(obj, getInner().getName());
  }
  
  /**
   * Add to monitoring wrapper layer.
   *
   * @param obj  the obj
   * @param name the name
   * @return the monitoring wrapper layer
   */
  public MonitoringWrapperLayer addTo(final MonitoredObject obj, final String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final AtomicLong passbackNanos = new AtomicLong(0);
    final NNResult[] wrappedInput = Arrays.stream(inObj).map(result -> new NNResult(result.getData(), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      passbackNanos.addAndGet(TimedResult.time(() -> result.accumulate(buffer, data)).timeNanos);
    }) {
    
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      
      @Override
      public boolean isAlive() {
        return result.isAlive();
      }
    }).toArray(i -> new NNResult[i]);
    TimedResult<NNResult> timedResult = TimedResult.time(() -> getInner().eval(wrappedInput));
    final NNResult output = timedResult.result;
    forwardPerformance.add((timedResult.timeNanos) / 1000000000.0);
    totalBatches++;
    Arrays.stream(inObj).forEach(x -> {
      x.addRef();
      x.getData().addRef();
    });
    final int items = Arrays.stream(inObj).mapToInt(x -> x.getData().length()).max().orElse(1);
    totalItems += items;
    if (recordSignalMetrics) {
      forwardSignal.clear();
      output.getData().stream().parallel().forEach(t -> {
        forwardSignal.add(t.getData());
      });
    }
    return new NNResult(output.getData(), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      if (recordSignalMetrics) {
        backwardSignal.clear();
        data.stream().parallel().forEach(t -> {
          backwardSignal.add(t.getData());
        });
      }
      backwardPerformance.add((TimedResult.time(() -> output.accumulate(buffer, data)).timeNanos - passbackNanos.getAndSet(0)) / (items * 1e9));
    }) {
  
      @Override
      protected void _free() {
        Arrays.stream(wrappedInput).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return output.isAlive();
      }
    };
  }
  
  /**
   * Gets backward performance.
   *
   * @return the backward performance
   */
  public PercentileStatistics getBackwardPerformance() {
    return backwardPerformance;
  }
  
  /**
   * Gets backward signal.
   *
   * @return the backward signal
   */
  public ScalarStatistics getBackwardSignal() {
    return backwardSignal;
  }
  
  /**
   * Gets forward performance.
   *
   * @return the forward performance
   */
  public PercentileStatistics getForwardPerformance() {
    return forwardPerformance;
  }
  
  /**
   * Gets forward signal.
   *
   * @return the forward signal
   */
  public ScalarStatistics getForwardSignal() {
    return forwardSignal;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    //json.fn("forwardPerf",forwardPerf.getJson());
    //json.fn("backwardPerf",backwardPerf.getJson());
    json.add("heapCopy", getInner().getJson(resources, dataSerializer));
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    json.addProperty("recordSignalMetrics", recordSignalMetrics);
    return json;
  }
  
  @Override
  public Map<String, Object> getMetrics() {
    final HashMap<String, Object> map = new HashMap<>();
    map.put("class", inner.getClass().getName());
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("outputStatistics", forwardSignal.getMetrics());
    map.put("backpropStatistics", backwardSignal.getMetrics());
    if (verbose) {
      map.put("forwardPerformance", forwardPerformance.getMetrics());
      map.put("backwardPerformance", backwardPerformance.getMetrics());
    }
    final double batchesPerItem = totalBatches * 1.0 / totalItems;
    map.put("avgMsPerItem", 1000 * batchesPerItem * forwardPerformance.getMean());
    map.put("medianMsPerItem", 1000 * batchesPerItem * forwardPerformance.getPercentile(0.5));
    final double backpropMean = backwardPerformance.getMean();
    final double backpropMedian = backwardPerformance.getPercentile(0.5);
    map.put("avgMsPerItem_Backward", 1000 * batchesPerItem * backpropMean);
    map.put("medianMsPerItem_Backward", 1000 * batchesPerItem * backpropMedian);
    final List<double[]> state = state();
    final ScalarStatistics statistics = new PercentileStatistics();
    for (final double[] s : state) {
      for (final double v : s) {
        statistics.add(v);
      }
    }
    if (statistics.getCount() > 0) {
      final HashMap<String, Object> weightStats = new HashMap<>();
      weightStats.put("buffers", state.size());
      weightStats.putAll(statistics.getMetrics());
      map.put("weights", weightStats);
    }
    return map;
  }
  
  @Override
  public String getName() {
    return getInner().getName();
  }
  
  /**
   * Record signal metrics boolean.
   *
   * @return the boolean
   */
  public boolean recordSignalMetrics() {
    return recordSignalMetrics;
  }
  
  @Override
  public NNLayer setName(final String name) {
    if (null != getInner()) {
      getInner().setName(name);
    }
    return this;
  }
  
  /**
   * Should record signal metrics monitoring wrapper layer.
   *
   * @param recordSignalMetrics the record signal metrics
   * @return the monitoring wrapper layer
   */
  public MonitoringWrapperLayer shouldRecordSignalMetrics(final boolean recordSignalMetrics) {
    this.recordSignalMetrics = recordSignalMetrics;
    return this;
  }
}
