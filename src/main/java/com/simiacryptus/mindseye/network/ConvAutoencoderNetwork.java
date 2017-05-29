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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.activation.DropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.GaussianNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.MaxDropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.ToeplitzSynapseLayer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeConditions;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.trainable.ConstL12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.L12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.util.ml.Tensor;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class ConvAutoencoderNetwork implements MonitoredItem {
  
  private final MonitoredObject metrics = new MonitoredObject();
  @Override
  public Map<String, Object> getMetrics() {
    return metrics.getMetrics();
  }
  
  public static class RecursiveBuilder implements MonitoredItem {
  
    @Override
    public Map<String, Object> getMetrics() {
      HashMap<String, Object> map = new HashMap<>();
      for(int i=0;i<layers.size();i++) {
        map.put("layer"+i,layers.get(i).getMetrics());
      }
      return map;
    }
  
    private final List<Tensor[]> representations = new ArrayList<>();
    private final List<int[]> dimensions = new ArrayList<>();
    private final List<ConvAutoencoderNetwork> layers = new ArrayList<>();
    
    public RecursiveBuilder(Tensor[] data) {
      representations.add(data);
      dimensions.add(data[0].getDims());
    }
  
    public ConvAutoencoderNetwork growLayer(int... dims) {
      return growLayer(layers.isEmpty()?100:0, 1, 100, dims);
    }
  
    public ConvAutoencoderNetwork growLayer(int pretrainingSize, int pretrainingMinutes, int maxIterations, int[] dims) {
      ConvAutoencoderNetwork newLayer = configure(newLayer(dimensions.get(dimensions.size() - 1), dims)).build();
      dimensions.add(dims);
      layers.add(newLayer);
      Tensor[] data = representations.get(representations.size() - 1);
      ArrayList<Tensor> list = new ArrayList<>(Arrays.asList(data));
      Collections.shuffle(list);
      if(pretrainingSize > 0) {
        Tensor[] pretrainingSet = list.subList(0, pretrainingSize).toArray(new Tensor[]{});
        configure(newLayer.newTrainer()).setMaxIterations(maxIterations).setTimeoutMinutes(pretrainingMinutes).train(pretrainingSet);
      }
      configure(newLayer.newTrainer()).train(data);
      representations.add(newLayer.encode(data));
      return newLayer;
    }
  
    public void tune() {
      configure(new ConvAutoencoderNetwork.TrainingParameters() {
        @Override
        protected TrainingMonitor wrap(TrainingMonitor monitor) {
          return new TrainingMonitor() {
            @Override
            public void log(String msg) {
              monitor.log(msg);
            }
      
            @Override
            public void onStepComplete(IterativeTrainer.Step currentPoint) {
              layers.forEach(layer->{
                layer.inputNoise.shuffle();
                layer.encodedNoise.shuffle();
              });
              monitor.onStepComplete(currentPoint);
            }
          };
        }
  
        @Override
        public SimpleLossNetwork getTrainingNetwork() {
          PipelineNetwork student = new PipelineNetwork();
          student.add(getEncoder());
          student.add(getDecoder());
          return new SimpleLossNetwork(student, new MeanSqLossLayer());
        }
      }).train(representations.get(0));
    }
    
    protected ConvAutoencoderNetwork.TrainingParameters configure(ConvAutoencoderNetwork.TrainingParameters trainingParameters) {
      return trainingParameters;
    }
    
    protected ConvAutoencoderNetwork.Builder configure(ConvAutoencoderNetwork.Builder builder) {
      return builder;
    }
    
    public NNLayer echo() {
      PipelineNetwork network = new PipelineNetwork();
      network.add(getEncoder());
      network.add(getDecoder());
      return network;
    }
    
    public NNLayer getEncoder() {
      PipelineNetwork network = new PipelineNetwork();
      for (int i = 0; i < layers.size(); i++) {
        network.add(layers.get(i).getEncoder());
      }
      return network;
    }
    
    public NNLayer getDecoder() {
      PipelineNetwork network = new PipelineNetwork();
      for (int i = layers.size() - 1; i >= 0; i--) {
        network.add(layers.get(i).getDecoder());
      }
      return network;
    }
    
    public List<ConvAutoencoderNetwork> getLayers() {
      return Collections.unmodifiableList(layers);
    }
  
    }
  
  public ConvAutoencoderNetwork.TrainingParameters newTrainer() {
    return new ConvAutoencoderNetwork.TrainingParameters() {
      @Override
      protected TrainingMonitor wrap(TrainingMonitor monitor) {
        return new TrainingMonitor() {
          @Override
          public void log(String msg) {
            monitor.log(msg);
          }
          
          @Override
          public void onStepComplete(IterativeTrainer.Step currentPoint) {
            inputNoise.shuffle();
            encodedNoise.shuffle();
            monitor.onStepComplete(currentPoint);
          }
        };
      }
      
      @Override
      public SimpleLossNetwork getTrainingNetwork() {
        PipelineNetwork student = new PipelineNetwork();
        student.add(encoder);
        student.add(decoder);
        return new SimpleLossNetwork(student, new MeanSqLossLayer());
      }
    };
  }
  
  public static ConvAutoencoderNetwork.Builder newLayer(int[] outerSize, int[] innerSize) {
    return new ConvAutoencoderNetwork.Builder(outerSize, innerSize);
  }
  
  private final int[] outerSize;
  private final int[] innerSize;
  private final GaussianNoiseLayer inputNoise;
  private final BiasLayer encoderBias;
  private final ToeplitzSynapseLayer encoderSynapse;
  private final MaxDropoutNoiseLayer encoderSubsample;
  private final ReLuActivationLayer encoderActivation;
  private final DropoutNoiseLayer encodedNoise;
  private final NNLayer decoderSynapse;
  private final BiasLayer decoderBias;
  private final ReLuActivationLayer decoderActivation;
  private final NNLayer encoder;
  private final NNLayer decoder;
  
  protected ConvAutoencoderNetwork(ConvAutoencoderNetwork.Builder networkParameters) {
    Random random = new Random();
    
    this.outerSize = networkParameters.getOuterSize();
    this.innerSize = networkParameters.getInnerSize();
    
    this.inputNoise = new GaussianNoiseLayer().setValue(networkParameters.getNoise());
    this.encoderSynapse = new ToeplitzSynapseLayer(this.outerSize, this.innerSize);
    this.encoderSubsample = new MaxDropoutNoiseLayer(2,2,1);
    this.encoderBias = new BiasLayer(this.innerSize).setWeights(i -> 0.0);
    this.encoderSynapse.setWeights(() -> random.nextGaussian() * 0.001);
    this.encoderActivation = new ReLuActivationLayer().freeze();
    this.encodedNoise = new DropoutNoiseLayer().setValue(networkParameters.getDropout());
    this.decoderSynapse = new ToeplitzSynapseLayer(this.innerSize, this.outerSize);
    this.decoderBias = new BiasLayer(this.outerSize).setWeights(i -> 0.0);
    this.decoderActivation = new ReLuActivationLayer().freeze();
  
    PipelineNetwork encoder = new PipelineNetwork();
    encoder.add(inputNoise);
    encoder.add(new MonitoringWrapper(encoderSynapse).addTo(metrics,"encoderSynapse"));
    encoder.add(encoderSubsample);
    encoder.add(encoderBias);
    encoder.add(encoderActivation);
    encoder.add(encodedNoise);
    this.encoder = new MonitoringWrapper(encoder).addTo(metrics,"encoder");
  
    PipelineNetwork decoder = new PipelineNetwork();
    decoder.add(new MonitoringWrapper(decoderSynapse).addTo(metrics,"decoderSynapse"));
    decoder.add(decoderBias);
    decoder.add(decoderActivation);
    this.decoder = new MonitoringWrapper(decoder).addTo(metrics,"decoder");
  }
  
  public Tensor[] encode(Tensor[] data) {
    return encoder.eval(NNResult.batchResultArray(Arrays.stream(data).map(x -> new Tensor[]{x}).toArray(i -> new Tensor[i][])))
               .data;
  }
  
  public int[] getOuterSize() {
    return outerSize;
  }
  
  public int[] getInnerSize() {
    return innerSize;
  }
  
  public GaussianNoiseLayer getInputNoise() {
    return inputNoise;
  }
  
  public BiasLayer getEncoderBias() {
    return encoderBias;
  }
  
  public NNLayer getEncoderSynapse() {
    return encoderSynapse;
  }
  
  public NNLayer getEncoderActivation() {
    return encoderActivation;
  }
  
  public DropoutNoiseLayer getEncodedNoise() {
    return encodedNoise;
  }
  
  public NNLayer getDecoderSynapse() {
    return decoderSynapse;
  }
  
  public BiasLayer getDecoderBias() {
    return decoderBias;
  }
  
  public NNLayer getDecoderActivation() {
    return decoderActivation;
  }
  
  public NNLayer getEncoder() {
    return encoder;
  }
  
  public NNLayer getDecoder() {
    return decoder;
  }
  
  public static abstract class TrainingParameters {
    private int sampleSize = Integer.MAX_VALUE;
    private double l1normalization = 0.0;
    private double l2normalization = 0.0;
    private OrientationStrategy orient = new LBFGS().setMinHistory(5).setMaxHistory(35);
    private LineSearchStrategy step = new ArmijoWolfeConditions().setC2(0.9).setAlpha(1e-4);
    private TrainingMonitor monitor = null;
    private int timeoutMinutes = 10;
    private double endFitness = Double.NEGATIVE_INFINITY;
    private int maxIterations = Integer.MAX_VALUE;
  
    public void train(Tensor... data) {
      SimpleLossNetwork trainingNetwork = getTrainingNetwork();
      StochasticArrayTrainable trainable = new StochasticArrayTrainable(Arrays.stream(data).map(x -> new Tensor[]{x, x}).toArray(i -> new Tensor[i][]), trainingNetwork, getSampleSize());
      L12Normalizer normalized = new ConstL12Normalizer(trainable).setFactor_L1(getL1normalization()).setFactor_L2(getL2normalization());
      IterativeTrainer trainer = new IterativeTrainer(normalized);
      trainer.setOrientation(getOrient());
      trainer.setScaling(getStep());
      TrainingMonitor monitor = getMonitor();
      trainer.setMonitor(wrap(monitor));
      trainer.setTimeout(getTimeoutMinutes(), TimeUnit.MINUTES);
      trainer.setTerminateThreshold(getEndFitness());
      trainer.setMaxIterations(maxIterations);
      trainer.run();
    }
    
    protected abstract TrainingMonitor wrap(TrainingMonitor monitor);
    
    public abstract SimpleLossNetwork getTrainingNetwork();
    
    public int getSampleSize() {
      return sampleSize;
    }
    
    public double getL1normalization() {
      return l1normalization;
    }
    
    public double getL2normalization() {
      return l2normalization;
    }
    
    public OrientationStrategy getOrient() {
      return orient;
    }
    
    public LineSearchStrategy getStep() {
      return step;
    }
    
    public TrainingMonitor getMonitor() {
      return monitor;
    }
    
    public int getTimeoutMinutes() {
      return timeoutMinutes;
    }
    
    public double getEndFitness() {
      return endFitness;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setSampleSize(int sampleSize) {
      this.sampleSize = sampleSize;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setL1normalization(double l1normalization) {
      this.l1normalization = l1normalization;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setL2normalization(double l2normalization) {
      this.l2normalization = l2normalization;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setOrient(OrientationStrategy orient) {
      this.orient = orient;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setStep(LineSearchStrategy step) {
      this.step = step;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setMonitor(TrainingMonitor monitor) {
      this.monitor = monitor;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setTimeoutMinutes(int timeoutMinutes) {
      this.timeoutMinutes = timeoutMinutes;
      return this;
    }
    
    public ConvAutoencoderNetwork.TrainingParameters setEndFitness(double endFitness) {
      this.endFitness = endFitness;
      return this;
    }
  
    public ConvAutoencoderNetwork.TrainingParameters setMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
    }
  
    public int getMaxIterations() {
      return maxIterations;
    }
  }
  
  public static class Builder {
    
    private final int[] outerSize;
    private final int[] innerSize;
    private double noise = 0.0;
    private double initRadius = 0.5;
    private int initStiffness = 3;
    private double initPeak = 0.001;
    private double dropout = 0.0;
    
    private Builder(int[] outerSize, int[] innerSize) {
      this.outerSize = outerSize;
      this.innerSize = innerSize;
    }
    
    public int[] getOuterSize() {
      return outerSize;
    }
    
    public int[] getInnerSize() {
      return innerSize;
    }
    
    public double getNoise() {
      return noise;
    }
    
    public double getInitRadius() {
      return initRadius;
    }
    
    public int getInitStiffness() {
      return initStiffness;
    }
    
    public double getInitPeak() {
      return initPeak;
    }
    
    public double getDropout() {
      return dropout;
    }
    
    public ConvAutoencoderNetwork.Builder setNoise(double noise) {
      this.noise = noise;
      return this;
    }
    
    public ConvAutoencoderNetwork.Builder setInitRadius(double initRadius) {
      this.initRadius = initRadius;
      return this;
    }
    
    public ConvAutoencoderNetwork.Builder setInitStiffness(int initStiffness) {
      this.initStiffness = initStiffness;
      return this;
    }
    
    public ConvAutoencoderNetwork.Builder setInitPeak(double initPeak) {
      this.initPeak = initPeak;
      return this;
    }
    
    public ConvAutoencoderNetwork.Builder setDropout(double dropout) {
      this.dropout = dropout;
      return this;
    }
    
    public ConvAutoencoderNetwork build() {
      return new ConvAutoencoderNetwork(ConvAutoencoderNetwork.Builder.this);
    }
  }
}
