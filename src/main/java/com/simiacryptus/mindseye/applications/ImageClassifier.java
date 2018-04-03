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

package com.simiacryptus.mindseye.applications;

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.ConstantResult;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.Explodable;
import com.simiacryptus.mindseye.layers.cudnn.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.models.NetworkFactory;
import com.simiacryptus.mindseye.network.DAGNetwork;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Image classifier.
 */
public abstract class ImageClassifier implements NetworkFactory {
  
  /**
   * The constant log.
   */
  protected static final Logger log = LoggerFactory.getLogger(ImageClassifier.class);
  /**
   * The Network.
   */
  protected volatile Layer cachedLayer;
  /**
   * The Prototype.
   */
  @Nullable
  protected
  Tensor prototype = new Tensor(224, 224, 3);
  /**
   * The Cnt.
   */
  protected int cnt = 1;
  /**
   * The Precision.
   */
  @Nonnull
  protected
  Precision precision = Precision.Float;
  private int batchSize;
  
  /**
   * Predict list.
   *
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param batchSize  the batch size
   * @param data       the data
   * @return the list
   */
  public static List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull Layer network, int count, @Nonnull List<CharSequence> categories, int batchSize, Tensor... data) {
    return predict(network, count, categories, batchSize, true, false, data);
  }
  
  /**
   * Predict list.
   *
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param batchSize  the batch size
   * @param asyncGC    the async gc
   * @param nullGC     the null gc
   * @param data       the data
   * @return the list
   */
  public static List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull Layer network, int count, @Nonnull List<CharSequence> categories, int batchSize, boolean asyncGC, boolean nullGC, Tensor[] data) {
    try {
      return Lists.partition(Arrays.asList(data), 1).stream().flatMap(batch -> {
        Tensor[][] input = {
          batch.stream().toArray(i -> new Tensor[i])
        };
        Result[] inputs = ConstantResult.singleResultArray(input);
        @Nullable Result result = network.eval(inputs);
        result.freeRef();
        TensorList resultData = result.getData();
        //Arrays.stream(input).flatMap(Arrays::stream).forEach(ReferenceCounting::freeRef);
        //Arrays.stream(inputs).forEach(ReferenceCounting::freeRef);
        //Arrays.stream(inputs).map(Result::getData).forEach(ReferenceCounting::freeRef);
  
        List<LinkedHashMap<CharSequence, Double>> maps = resultData.stream().map(tensor -> {
          @Nullable double[] predictionSignal = tensor.getData();
          int[] order = IntStream.range(0, 1000).mapToObj(x -> x)
            .sorted(Comparator.comparing(i -> -predictionSignal[i]))
            .mapToInt(x -> x).toArray();
          assert categories.size() == predictionSignal.length;
          @Nonnull LinkedHashMap<CharSequence, Double> topN = new LinkedHashMap<>();
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
   * @param network the network
   * @return the training monitor
   */
  @Nonnull
  public static TrainingMonitor getTrainingMonitor(@Nonnull ArrayList<StepRecord> history, final PipelineNetwork network) {
    return TestUtil.getMonitor(history);
  }
  
  /**
   * Add.
   *
   * @param layer the layer
   * @param model the model
   * @return the layer
   */
  @Nonnull
  protected static Layer add(@Nonnull Layer layer, @Nonnull PipelineNetwork model) {
    name(layer);
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          ((DAGNetwork) explode).visitNodes(node -> name(node.getLayer()));
          log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), ((DAGNetwork) explode).getNodes().size()));
        }
        else {
          log.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getName()));
        }
        return add(explode, model);
      } finally {
        layer.freeRef();
      }
    }
    else {
      model.wrap(layer);
      return layer;
    }
  }
  
  /**
   * Evaluate prototype tensor.
   *
   * @param layer         the layer
   * @param prevPrototype the prev prototype
   * @param cnt           the cnt
   * @return the tensor
   */
  @Nonnull
  protected static Tensor evaluatePrototype(@Nonnull final Layer layer, final Tensor prevPrototype, int cnt) {
    int numberOfParameters = layer.state().stream().mapToInt(x -> x.length).sum();
    @Nonnull int[] prev_dimensions = prevPrototype.getDimensions();
    Result eval = layer.eval(prevPrototype);
    TensorList newPrototype = eval.getData();
    if (null != prevPrototype) prevPrototype.freeRef();
    eval.freeRef();
    try {
      @Nonnull int[] new_dimensions = newPrototype.getDimensions();
      log.info(String.format("Added layer #%d: %s; %s params, dimensions %s (%s) -> %s (%s)", //
        cnt, layer, numberOfParameters, //
        Arrays.toString(prev_dimensions), Tensor.length(prev_dimensions), //
        Arrays.toString(new_dimensions), Tensor.length(new_dimensions)));
      return newPrototype.get(0);
    } finally {
      newPrototype.freeRef();
    }
  }
  
  /**
   * Name.
   *
   * @param layer the layer
   */
  protected static void name(final Layer layer) {
    if (layer.getName().contains(layer.getId().toString())) {
      if (layer instanceof ConvolutionLayer) {
        layer.setName(layer.getClass().getSimpleName() + ((ConvolutionLayer) layer).getConvolutionParams());
      }
      else if (layer instanceof SimpleConvolutionLayer) {
        layer.setName(String.format("%s: %s", layer.getClass().getSimpleName(),
          Arrays.toString(((SimpleConvolutionLayer) layer).getKernelDimensions())));
      }
      else if (layer instanceof FullyConnectedLayer) {
        layer.setName(String.format("%s:%sx%s",
          layer.getClass().getSimpleName(),
          Arrays.toString(((FullyConnectedLayer) layer).inputDims),
          Arrays.toString(((FullyConnectedLayer) layer).outputDims)));
      }
      else if (layer instanceof BiasLayer) {
        layer.setName(String.format("%s:%s",
          layer.getClass().getSimpleName(),
          ((BiasLayer) layer).bias.length));
      }
    }
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
//      CudaTensorList gpuInput = CudnnHandle.apply(gpu -> {
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
   * Predict list.
   *
   * @param network    the network
   * @param count      the count
   * @param categories the categories
   * @param data       the data
   * @return the list
   */
  public List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull Layer network, int count, @Nonnull List<CharSequence> categories, @Nonnull Tensor... data) {
    return predict(network, count, categories, Math.max(data.length, getBatchSize()), data);
  }
  
  /**
   * Gets categories.
   *
   * @return the categories
   */
  public abstract List<CharSequence> getCategories();
  
  /**
   * Predict list.
   *
   * @param count the count
   * @param data  the data
   * @return the list
   */
  public List<LinkedHashMap<CharSequence, Double>> predict(int count, Tensor... data) {
    return predict(getNetwork(), count, getCategories(), data);
  }
  
  /**
   * Predict list.
   *
   * @param network the network
   * @param count   the count
   * @param data    the data
   * @return the list
   */
  public List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull Layer network, int count, Tensor[] data) {
    return predict(network, count, getCategories(), data);
  }
  
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
  @Nonnull
  public ImageClassifier setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
  
  /**
   * Deep dream.
   *
   * @param log                 the log
   * @param image               the image
   * @param targetCategoryIndex the target category index
   * @param totalCategories     the total categories
   * @param config              the config
   */
  public void deepDream(@Nonnull final NotebookOutput log, final Tensor image, final int targetCategoryIndex, final int totalCategories, Function<IterativeTrainer, IterativeTrainer> config) {deepDream(log, image, targetCategoryIndex, totalCategories, config, getNetwork(), new EntropyLossLayer(), -1.0);}
  
  /**
   * Sets precision.
   *
   * @param model     the model
   * @param precision the precision
   */
  public static void setPrecision(DAGNetwork model, final Precision precision) {
    model.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }
  
  @Nonnull
  @Override
  public Layer getNetwork() {
    if (null == cachedLayer) {
      synchronized (this) {
        if (null == cachedLayer) {
          try {
            cachedLayer = buildNetwork();
            setPrecision((DAGNetwork) cachedLayer);
            if (null != prototype) prototype.freeRef();
            prototype = null;
            return cachedLayer;
          } catch (@Nonnull final RuntimeException e) {
            throw e;
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        }
      }
    }
    return cachedLayer;
    
    
  }
  
  /**
   * Build network layer.
   *
   * @return the layer
   */
  protected abstract Layer buildNetwork();
  
  /**
   * Deep dream.
   *
   * @param log                 the log
   * @param image               the image
   * @param targetCategoryIndex the target category index
   * @param totalCategories     the total categories
   * @param config              the config
   * @param network             the network
   * @param lossLayer           the loss layer
   * @param targetValue         the target value
   */
  public void deepDream(@Nonnull final NotebookOutput log, final Tensor image, final int targetCategoryIndex, final int totalCategories, Function<IterativeTrainer, IterativeTrainer> config, final Layer network, final Layer lossLayer, final double targetValue) {
    @Nonnull List<Tensor[]> data = Arrays.<Tensor[]>asList(new Tensor[]{
      image, new Tensor(1, 1, totalCategories).set(targetCategoryIndex, targetValue)
    });
    log.code(() -> {
      for (Tensor[] tensors : data) {
        ImageClassifier.log.info(log.image(tensors[0].toImage(), "") + tensors[1]);
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
      supervised.wrap(lossLayer,
        supervised.add(network.freeze(),
          supervised.wrap(clamp, supervised.getInput(0))),
        supervised.getInput(1));
//      TensorList[] gpuInput = data.stream().map(data1 -> {
//        return CudnnHandle.apply(gpu -> {
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
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
      return TestUtil.plot(history);
    });
  }
  
  /**
   * Sets precision.
   *
   * @param model the model
   */
  protected void setPrecision(DAGNetwork model) {setPrecision(model, precision);}
  
}
