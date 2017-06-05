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
import com.simiacryptus.mindseye.layers.reducers.ImgConcatLayer;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.ScalarStatistics;
import com.simiacryptus.util.ml.Tensor;

import java.util.*;

@SuppressWarnings("serial")
public final class MonitoringSynapse extends NNLayer implements MonitoredItem {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches",totalBatches);
    json.addProperty("totalItems",totalItems);
    json.addProperty("enabled",enabled);
    return json;
  }
  public static MonitoringSynapse fromJson(JsonObject json) {
    MonitoringSynapse obj = new MonitoringSynapse(json);
    obj.totalBatches = json.get("totalBatches").getAsInt();
    obj.totalItems = json.get("totalItems").getAsInt();
    obj.enabled = json.get("enabled").getAsBoolean();
    obj.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    obj.forwardStatistics.readJson(json.getAsJsonObject("forwardStatistics"));
    return obj;
  }
  protected MonitoringSynapse(JsonObject id) {
    super(id);
  }
  
  private int totalBatches = 0;
  private int totalItems = 0;
  private final ScalarStatistics backpropStatistics = new ScalarStatistics();
  private final ScalarStatistics forwardStatistics = new ScalarStatistics();
  boolean enabled = false;
  
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
        for(Tensor t : data) {
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
