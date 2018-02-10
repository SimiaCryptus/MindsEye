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

package com.simiacryptus.mindseye.models;

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.lang.NNConstant;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.GpuSystem;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Image classifier.
 */
public abstract class ImageClassifier {
  private int batchSize;
  private volatile NNLayer network;
  
  /**
   * Predict list.
   *
   * @param prefilter  the prefilter
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param batchSize  the batch size
   * @param data       the data
   * @return the list
   */
  public static List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull NNLayer network, int count, @javax.annotation.Nonnull List<String> categories, int batchSize, Tensor... data) {
    return predict(prefilter, network, count, categories, batchSize, true, false, data);
  }
  
  /**
   * Predict list.
   *
   * @param prefilter  the prefilter
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param batchSize  the batch size
   * @param asyncGC    the async gc
   * @param nullGC     the null gc
   * @param data       the data
   * @return the list
   */
  public static List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull NNLayer network, int count, @javax.annotation.Nonnull List<String> categories, int batchSize, boolean asyncGC, boolean nullGC, Tensor[] data) {
    @javax.annotation.Nonnull Executor garbageman = (!nullGC && asyncGC) ? Executors.newSingleThreadExecutor() : command -> {
      if (!nullGC) command.run();
    };
    try {
      return Lists.partition(Arrays.asList(data), batchSize).stream().flatMap(batch -> {
        @javax.annotation.Nullable NNResult nnResult = network.eval(NNConstant.singleResultArray(new Tensor[][]{
          batch.stream().map(prefilter).toArray(i -> new Tensor[i])
        }));
        List<Tensor> tensorList = nnResult.getData().stream().collect(Collectors.toList());
        garbageman.execute(GpuSystem::cleanMemory);
        return tensorList.stream().map(tensor -> {
          @Nullable double[] predictionSignal = tensor.getData();
          int[] order = IntStream.range(0, 1000).mapToObj(x -> x)
            .sorted(Comparator.comparing(i -> -predictionSignal[i]))
            .mapToInt(x -> x).toArray();
          assert categories.size() == predictionSignal.length;
          @javax.annotation.Nonnull LinkedHashMap<String, Double> topN = new LinkedHashMap<>();
          for (int i = 0; i < count; i++) {
            int index = order[i];
            topN.put(categories.get(index), predictionSignal[index]);
          }
          tensor.freeRef();
          return topN;
        });
      }).collect(Collectors.toList());
    } finally {
      if (garbageman instanceof ExecutorService) {((ExecutorService) garbageman).shutdown();}
    }
  }
  
  /**
   * Predict list.
   *
   * @param prefilter  the prefilter
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param data       the data
   * @return the list
   */
  public List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull NNLayer network, int count, @javax.annotation.Nonnull List<String> categories, @javax.annotation.Nonnull Tensor... data) {
    return predict(prefilter, network, count, categories, Math.max(data.length, getBatchSize()), data);
  }
  
  /**
   * Prefilter tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  @javax.annotation.Nonnull
  public abstract Tensor prefilter(Tensor tensor);
  
  /**
   * Gets categories.
   *
   * @return the categories
   */
  public abstract List<String> getCategories();
  
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
  
  /**
   * Predict list.
   *
   * @param network the network
   * @param count   the count
   * @param data    the data
   * @return the list
   */
  public List<LinkedHashMap<String, Double>> predict(@javax.annotation.Nonnull NNLayer network, int count, Tensor[] data) {
    return predict(this::prefilter, network, count, getCategories(), data);
  }
  
  /**
   * Gets network.
   *
   * @return the network
   */
  public abstract NNLayer getNetwork();
  
  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }
  
  /**
   * Sets batch size.
   *
   * @param batchSize the batch size
   * @return the batch size
   */
  @javax.annotation.Nonnull
  public ImageClassifier setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
}
