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

import javax.annotation.Nullable;
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
  protected MonitoringWrapperLayer(@javax.annotation.Nonnull final JsonObject json) {
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
  public MonitoringWrapperLayer(final Layer inner) {
    super(inner);
  }
  
  /**
   * From json monitoring wrapper layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the monitoring wrapper layer
   */
  public static MonitoringWrapperLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new MonitoringWrapperLayer(json);
  }
  
  /**
   * Add to monitoring wrapper layer.
   *
   * @param obj the obj
   * @return the monitoring wrapper layer
   */
  @javax.annotation.Nonnull
  public MonitoringWrapperLayer addTo(@javax.annotation.Nonnull final MonitoredObject obj) {
    return addTo(obj, getInner().getName());
  }
  
  /**
   * Add to monitoring wrapper layer.
   *
   * @param obj  the obj
   * @param name the name
   * @return the monitoring wrapper layer
   */
  @javax.annotation.Nonnull
  public MonitoringWrapperLayer addTo(@javax.annotation.Nonnull final MonitoredObject obj, final String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }
  
  @Override
  public Result evalAndFree(@javax.annotation.Nonnull final Result... inObj) {
    @javax.annotation.Nonnull final AtomicLong passbackNanos = new AtomicLong(0);
    final Result[] wrappedInput = Arrays.stream(inObj).map(result -> {
      return new Result(result.getData(), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
        passbackNanos.addAndGet(TimedResult.time(() -> result.accumulate(buffer, data)).timeNanos);
      }) {
  
        @Override
        protected void _free() {
          result.freeRef();
        }
  
  
        @Override
        public boolean isAlive() {
          return result.isAlive();
        }
      };
    }).toArray(i -> new Result[i]);
    @javax.annotation.Nonnull TimedResult<Result> timedResult = TimedResult.time(() -> getInner().evalAndFree(wrappedInput));
    final Result output = timedResult.result;
    forwardPerformance.add((timedResult.timeNanos) / 1000000000.0);
    totalBatches++;
    final int items = Arrays.stream(inObj).mapToInt(x -> x.getData().length()).max().orElse(1);
    totalItems += items;
    if (recordSignalMetrics) {
      forwardSignal.clear();
      output.getData().stream().parallel().forEach(t -> {
        forwardSignal.add(t.getData());
        t.freeRef();
      });
    }
    return new Result(output.getData(), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (recordSignalMetrics) {
        backwardSignal.clear();
        data.stream().parallel().forEach(t -> {
          backwardSignal.add(t.getData());
          t.freeRef();
        });
      }
      backwardPerformance.add((TimedResult.time(() -> output.accumulate(buffer, data)).timeNanos - passbackNanos.getAndSet(0)) / (items * 1e9));
    }) {
  
      @Override
      protected void _free() {
        output.freeRef();
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
  @javax.annotation.Nonnull
  public PercentileStatistics getBackwardPerformance() {
    return backwardPerformance;
  }
  
  /**
   * Gets backward signal.
   *
   * @return the backward signal
   */
  @javax.annotation.Nonnull
  public ScalarStatistics getBackwardSignal() {
    return backwardSignal;
  }
  
  /**
   * Gets forward performance.
   *
   * @return the forward performance
   */
  @javax.annotation.Nonnull
  public PercentileStatistics getForwardPerformance() {
    return forwardPerformance;
  }
  
  /**
   * Gets forward signal.
   *
   * @return the forward signal
   */
  @javax.annotation.Nonnull
  public ScalarStatistics getForwardSignal() {
    return forwardSignal;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    //json.fn("forwardPerf",forwardPerf.getJson());
    //json.fn("backwardPerf",backwardPerf.getJson());
    json.add("heapCopy", getInner().getJson(resources, dataSerializer));
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    json.addProperty("recordSignalMetrics", recordSignalMetrics);
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Map<String, Object> getMetrics() {
    @javax.annotation.Nonnull final HashMap<String, Object> map = new HashMap<>();
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
    @Nullable final List<double[]> state = state();
    @javax.annotation.Nonnull final ScalarStatistics statistics = new PercentileStatistics();
    for (@javax.annotation.Nonnull final double[] s : state) {
      for (final double v : s) {
        statistics.add(v);
      }
    }
    if (statistics.getCount() > 0) {
      @javax.annotation.Nonnull final HashMap<String, Object> weightStats = new HashMap<>();
      weightStats.put("buffers", state.size());
      weightStats.putAll(statistics.getMetrics());
      map.put("weights", weightStats);
    }
    return map;
  }
  
  @javax.annotation.Nullable
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
  
  @javax.annotation.Nonnull
  @Override
  public Layer setName(final String name) {
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
  @javax.annotation.Nonnull
  public MonitoringWrapperLayer shouldRecordSignalMetrics(final boolean recordSignalMetrics) {
    this.recordSignalMetrics = recordSignalMetrics;
    return this;
  }
}
