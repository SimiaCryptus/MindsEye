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

/**
 * The type Monitoring synapse.
 */
@SuppressWarnings("serial")
public final class MonitoringSynapse extends NNLayer implements MonitoredItem {
  
  private final ScalarStatistics backpropStatistics = new PercentileStatistics();
  private final ScalarStatistics forwardStatistics = new PercentileStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;
  
  /**
   * Instantiates a new Monitoring synapse.
   *
   * @param id the id
   */
  protected MonitoringSynapse(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Monitoring synapse.
   */
  public MonitoringSynapse() {
    super();
  }
  
  /**
   * From json monitoring synapse.
   *
   * @param json the json
   * @return the monitoring synapse
   */
  public static MonitoringSynapse fromJson(JsonObject json) {
    MonitoringSynapse obj = new MonitoringSynapse(json);
    obj.totalBatches = json.get("totalBatches").getAsInt();
    obj.totalItems = json.get("totalItems").getAsInt();
    obj.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    obj.forwardStatistics.readJson(json.getAsJsonObject("forwardStatistics"));
    return obj;
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    assert (1 == inObj.length);
    NNResult input = inObj[0];
    long start = System.nanoTime();
    double elapsed = (System.nanoTime() - start) / 1000000000.0;
    totalBatches++;
    totalItems += input.getData().length();
    forwardStatistics.clear();
    input.getData().stream().parallel().forEach(t -> {
      forwardStatistics.add(t.getData());
    });
    return new NNResult(input.getData()) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        backpropStatistics.clear();
        data.stream().parallel().forEach(t -> {
          backpropStatistics.add(t.getData());
        });
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
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forward", forwardStatistics.getMetrics());
    map.put("backprop", backpropStatistics.getMetrics());
    return map;
  }
  
  /**
   * Add to monitoring synapse.
   *
   * @param obj the obj
   * @return the monitoring synapse
   */
  public MonitoringSynapse addTo(MonitoredObject obj) {
    return addTo(obj, getName());
  }
  
  /**
   * Add to monitoring synapse.
   *
   * @param obj  the obj
   * @param name the name
   * @return the monitoring synapse
   */
  public MonitoringSynapse addTo(MonitoredObject obj, String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }
}
