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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A diagnostic pass-through layer that collects value statistics of forward and backprop signals.
 */
@SuppressWarnings("serial")
public final class MonitoringSynapse extends LayerBase implements MonitoredItem {
  
  private final ScalarStatistics backpropStatistics = new PercentileStatistics();
  private final ScalarStatistics forwardStatistics = new PercentileStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;
  
  /**
   * Instantiates a new Monitoring synapse.
   */
  public MonitoringSynapse() {
    super();
  }
  
  /**
   * Instantiates a new Monitoring synapse.
   *
   * @param id the id
   */
  protected MonitoringSynapse(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json monitoring synapse.
   *
   * @param json the json
   * @param rs   the rs
   * @return the monitoring synapse
   */
  @javax.annotation.Nonnull
  public static MonitoringSynapse fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    @javax.annotation.Nonnull final MonitoringSynapse obj = new MonitoringSynapse(json);
    obj.totalBatches = json.get("totalBatches").getAsInt();
    obj.totalItems = json.get("totalItems").getAsInt();
    obj.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    obj.forwardStatistics.readJson(json.getAsJsonObject("forwardStatistics"));
    return obj;
  }
  
  /**
   * Add to monitoring synapse.
   *
   * @param obj the obj
   * @return the monitoring synapse
   */
  @javax.annotation.Nonnull
  public MonitoringSynapse addTo(@javax.annotation.Nonnull final MonitoredObject obj) {
    return addTo(obj, getName());
  }
  
  /**
   * Add to monitoring synapse.
   *
   * @param obj  the obj
   * @param name the name
   * @return the monitoring synapse
   */
  @javax.annotation.Nonnull
  public MonitoringSynapse addTo(@javax.annotation.Nonnull final MonitoredObject obj, final String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }
  
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputdata = input.getData();
    input.addRef();
    inputdata.addRef();
    System.nanoTime();
    System.nanoTime();
    totalBatches++;
    totalItems += inputdata.length();
    forwardStatistics.clear();
    inputdata.stream().parallel().forEach(t -> {
      forwardStatistics.add(t.getData());
      t.freeRef();
    });
    return new Result(inputdata, (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      backpropStatistics.clear();
      data.addRef();
      input.accumulate(buffer, data);
      data.stream().parallel().forEach(t -> {
        backpropStatistics.add(t.getData());
        t.freeRef();
      });
    }) {
      
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
  
      @Override
      protected void _free() {
        input.freeRef();
      }
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Map<String, Object> getMetrics() {
    @javax.annotation.Nonnull final HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forward", forwardStatistics.getMetrics());
    map.put("backprop", backpropStatistics.getMetrics());
    return map;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
