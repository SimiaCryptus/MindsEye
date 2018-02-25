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
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Image classifier.
 */
public abstract class ImageClassifier implements NetworkFactory {
  private static final Logger logger = LoggerFactory.getLogger(ReferenceCountingBase.class);
  private int batchSize;
  private volatile Layer network;
  
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
  public static List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull Layer network, int count, @javax.annotation.Nonnull List<String> categories, int batchSize, Tensor... data) {
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
  public static List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull Layer network, int count, @javax.annotation.Nonnull List<String> categories, int batchSize, boolean asyncGC, boolean nullGC, Tensor[] data) {
    try {
      return Lists.partition(Arrays.asList(data), 1).stream().flatMap(batch -> {
        Tensor[][] input = {
          batch.stream().map(prefilter).toArray(i -> new Tensor[i])
        };
        Result[] inputs = ConstantResult.singleResultArray(input);
        @Nullable Result result = network.eval(inputs);
        result.freeRef();
        TensorList resultData = result.getData();
        Arrays.stream(input).flatMap(Arrays::stream).forEach(ReferenceCounting::freeRef);
        Arrays.stream(inputs).forEach(ReferenceCounting::freeRef);
        Arrays.stream(inputs).map(Result::getData).forEach(ReferenceCounting::freeRef);
  
        List<LinkedHashMap<String, Double>> maps = resultData.stream().map(tensor -> {
          @Nullable double[] predictionSignal = tensor.getData();
          int[] order = IntStream.range(0, 1000).mapToObj(x -> x)
            .sorted(Comparator.comparing(i -> -predictionSignal[i]))
            .mapToInt(x -> x).toArray();
          assert categories.size() == predictionSignal.length;
          @Nonnull LinkedHashMap<String, Double> topN = new LinkedHashMap<>();
          for (int i = 0; i < count; i++) {
            int index = order[i];
            topN.put(categories.get(index), predictionSignal[index]);
          }
          tensor.freeRef();
          return topN;
        }).collect(Collectors.toList());
        resultData.freeRef();
        return maps.stream();
      }).collect(Collectors.toList());
    } finally {
    }
  }
  
  /**
   * Gets training monitor.
   *
   * @param history the history
   * @param network
   * @return the training monitor
   */
  @javax.annotation.Nonnull
  public static TrainingMonitor getTrainingMonitor(@Nonnull ArrayList<StepRecord> history, final PipelineNetwork network) {
    return TestUtil.getMonitor(history);
  }
  
  /**
   * Deep dream.
   *
   * @param log   the log
   * @param image the image
   */
  public void deepDream(@Nonnull final NotebookOutput log, final Tensor image) {
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      @Nonnull PipelineNetwork supervised = new PipelineNetwork(1);
      supervised.add(getNetwork().freeze(), supervised.wrap(clamp, supervised.getInput(0)));
//      CudaTensorList gpuInput = CudnnHandle.eval(gpu -> {
//        Precision precision = Precision.Float;
//        return CudaTensorList.wrap(gpu.getPtr(TensorArray.wrap(image), precision, MemoryType.Managed), 1, image.getDimensions(), precision);
//      });
//      @Nonnull Trainable trainable = new TensorListTrainable(supervised, gpuInput).setVerbosity(1).setMask(true);
      @Nonnull Trainable trainable = new ArrayTrainable(supervised, 1).setVerbose(true).setMask(true, false).setData(Arrays.<Tensor[]>asList(new Tensor[]{image}));
      new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history, supervised))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(60, TimeUnit.MINUTES)
        .runAndFree();
      return TestUtil.plot(history);
    });
  }
  
  /**
   * Deep dream.
   *
   * @param log                 the log
   * @param image               the image
   * @param targetCategoryIndex the target category index
   * @param totalCategories     the total categories
   */
  public void deepDream(@Nonnull final NotebookOutput log, final Tensor image, final int targetCategoryIndex, final int totalCategories, Function<IterativeTrainer, IterativeTrainer> config) {
    @Nonnull List<Tensor[]> data = Arrays.<Tensor[]>asList(new Tensor[]{
      image, new Tensor(totalCategories).set(targetCategoryIndex, 1.0)
    });
    log.code(() -> {
      for (Tensor[] tensors : data) {
        try {
          logger.info(log.image(tensors[0].toImage(), "") + tensors[1]);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    });
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      @Nonnull PipelineNetwork supervised = new PipelineNetwork(2);
      supervised.wrap(new EntropyLossLayer(),
        supervised.add(getNetwork().freeze(),
          supervised.wrap(clamp, supervised.getInput(0))),
        supervised.getInput(1));
//      TensorList[] gpuInput = data.stream().map(data1 -> {
//        return CudnnHandle.eval(gpu -> {
//          Precision precision = Precision.Float;
//          return CudaTensorList.wrap(gpu.getPtr(TensorArray.wrap(data1), precision, MemoryType.Managed), 1, image.getDimensions(), precision);
//        });
//      }).toArray(i -> new TensorList[i]);
//      @Nonnull Trainable trainable = new TensorListTrainable(supervised, gpuInput).setVerbosity(1).setMask(true);
      @Nonnull Trainable trainable = new ArrayTrainable(supervised, 1).setVerbose(true).setMask(true, false).setData(data);
      config.apply(new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history, supervised))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(60, TimeUnit.MINUTES))
        .runAndFree();
      return TestUtil.plot(history);
    });
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
  public List<LinkedHashMap<String, Double>> predict(Function<Tensor, Tensor> prefilter, @javax.annotation.Nonnull Layer network, int count, @javax.annotation.Nonnull List<String> categories, @javax.annotation.Nonnull Tensor... data) {
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
  public List<LinkedHashMap<String, Double>> predict(@javax.annotation.Nonnull Layer network, int count, Tensor[] data) {
    return predict(this::prefilter, network, count, getCategories(), data);
  }
  
  /**
   * Gets network.
   *
   * @return the network
   */
  public abstract Layer getNetwork();
  
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
