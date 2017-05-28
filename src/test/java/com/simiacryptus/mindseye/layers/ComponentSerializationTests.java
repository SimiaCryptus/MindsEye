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

package com.simiacryptus.mindseye.layers;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.activation.*;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumReducerLayer;
import com.simiacryptus.mindseye.layers.synapse.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class ComponentSerializationTests {
  
  private static final Logger log = LoggerFactory.getLogger(ComponentSerializationTests.class);
  
  @Test
  public void testPipeline() throws Throwable {
  
    Random random = new Random();
    PipelineNetwork network = new PipelineNetwork();
    network.add(new DenseSynapseLayer(new int[]{2,2},new int[]{2,2}).setWeights(() -> random.nextDouble()));
    network.add(new DenseSynapseLayer(new int[]{2,2},new int[]{2,2}).setWeights(() -> random.nextDouble()));
  
    JsonObject json = network.getJson();
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(json));
    NNLayer copy = NNLayer.fromJson(json);
  
    Tensor input = new Tensor(2, 2).fill(() -> random.nextDouble());
    NNResult a = network.eval(input);
    NNResult b = copy.eval(input);
    Assert.assertArrayEquals(a.data[0].getData(),b.data[0].getData(), 1e-8);
  
  
  }
  
}
