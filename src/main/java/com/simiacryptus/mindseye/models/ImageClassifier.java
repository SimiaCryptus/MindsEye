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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.io.NullNotebookOutput;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Image classifier.
 */
public abstract class ImageClassifier {
  private volatile NNLayer network;
  
  /**
   * Prefilter tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  public abstract Tensor prefilter(Tensor tensor);
  
  /**
   * Build pipeline network.
   *
   * @return the pipeline network
   */
  public PipelineNetwork build() {return build(new NullNotebookOutput());}
  
  /**
   * Build pipeline network.
   *
   * @param output the output
   * @return the pipeline network
   */
  public abstract PipelineNetwork build(NotebookOutput output);
  
  /**
   * Gets categories.
   *
   * @return the categories
   */
  public abstract List<String> getCategories();
  
  public static List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, NNLayer network, int count, List<String> categories, Tensor[] data) {
    return GpuController.call(ctx -> {
      TensorList evalResult = network.eval(ctx, NNResult.singleResultArray(new Tensor[][]{
        Arrays.stream(data).map(prefilter).toArray(i -> new Tensor[i])
      })).getData();
      return evalResult.stream().collect(Collectors.toList());
    }).stream().map(tensor -> {
      double[] predictionSignal = tensor.getData();
      int[] order = IntStream.range(0, 1000).mapToObj(x -> x)
                             .sorted(Comparator.comparing(i -> -predictionSignal[i]))
                             .mapToInt(x -> x).toArray();
      assert categories.size() == predictionSignal.length;
      LinkedHashMap<String, Double> topN = new LinkedHashMap<>();
      for (int i = 0; i < count; i++) {
        int index = order[i];
        topN.put(categories.get(index), predictionSignal[index]);
      }
      return topN;
    }).collect(Collectors.toList());
  }
  
  /**
   * Predict list.
   *
   * @param count the count
   * @param data  the data
   * @return the list
   */
  public List<LinkedHashMap<String, Double>> predict(int count, Tensor... data) {
    return predict(this::prefilter, getNetwork(), count, getCategories(), data);
  }
  
  public List<LinkedHashMap<String, Double>> predict(NNLayer network, int count, Tensor[] data) {
    return predict(this::prefilter, network, count, getCategories(), data);
  }
  
  /**
   * Gets network.
   *
   * @return the network
   */
  public NNLayer getNetwork() {
    if (null == network) {
      synchronized (this) {
        if (null == network) {
          network = build();
        }
      }
    }
    return network;
  }
}
