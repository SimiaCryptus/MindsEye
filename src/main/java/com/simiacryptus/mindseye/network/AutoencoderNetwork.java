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
import com.simiacryptus.mindseye.layers.activation.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.TransposedSynapseLayer;
import com.simiacryptus.mindseye.layers.util.VariableLayer;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeConditions;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.trainable.ConstL12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.L12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.util.ml.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class AutoencoderNetwork {
  
  public static class RecursiveBuilder {
    
    private final List<Tensor[]> representations = new ArrayList<>();
    private final List<int[]> dimensions = new ArrayList<>();
    private final List<AutoencoderNetwork> layers = new ArrayList<>();
    
    public RecursiveBuilder(Tensor[] data) {
      representations.add(data);
      dimensions.add(data[0].getDims());
    }
  
    public void trainingMode() {
      layers.forEach(x->x.trainingMode());
    }
  
    public void runMode() {
      layers.forEach(x->x.runMode());
    }
  
    public AutoencoderNetwork growLayer(int... dims) {
      return growLayer(layers.isEmpty()?100:0, 1, 10, dims);
    }
  
    public AutoencoderNetwork growLayer(int pretrainingSize, int pretrainingMinutes, int pretrainIterations, int[] dims) {
      trainingMode();
      AutoencoderNetwork newLayer = configure(newLayer(dimensions.get(dimensions.size() - 1), dims)).build();

      Tensor[] data = representations.get(representations.size() - 1);
      dimensions.add(dims);
      layers.add(newLayer);

      if(pretrainingSize > 0 && pretrainIterations > 0 && pretrainingMinutes > 0) {
        ArrayList<Tensor> list = new ArrayList<>(Arrays.asList(data));
        Collections.shuffle(list);
        Tensor[] pretrainingSet = list.subList(0, pretrainingSize).toArray(new Tensor[]{});
        configure(newLayer.train()).setMaxIterations(pretrainIterations).setTimeoutMinutes(pretrainingMinutes).run(pretrainingSet);
      }
      newLayer.decoderSynapse = ((TransposedSynapseLayer)newLayer.decoderSynapse).asNewSynapseLayer();
      newLayer.decoderSynapsePlaceholder.setInner(newLayer.decoderSynapse);
      configure(newLayer.train()).run(data);

      runMode();
      representations.add(newLayer.encode(data));
      return newLayer;
    }
  
    public void tune() {
      configure(new AutoencoderNetwork.TrainingParameters() {
        @Override
        protected TrainingMonitor wrap(TrainingMonitor monitor) {
          return new TrainingMonitor() {
            @Override
            public void log(String msg) {
              monitor.log(msg);
            }
      
            @Override
            public void onStepComplete(Step currentPoint) {
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
      }).run(representations.get(0));
    }
    
    protected AutoencoderNetwork.TrainingParameters configure(AutoencoderNetwork.TrainingParameters trainingParameters) {
      return trainingParameters;
    }
    
    protected AutoencoderNetwork.Builder configure(AutoencoderNetwork.Builder builder) {
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
    
    public List<AutoencoderNetwork> getLayers() {
      return Collections.unmodifiableList(layers);
    }
  }
  
  public AutoencoderNetwork.TrainingParameters train() {
    return new AutoencoderNetwork.TrainingParameters() {
      @Override
      protected TrainingMonitor wrap(TrainingMonitor monitor) {
        return new TrainingMonitor() {
          @Override
          public void log(String msg) {
            monitor.log(msg);
          }
          
          @Override
          public void onStepComplete(Step currentPoint) {
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
  
  public static AutoencoderNetwork.Builder newLayer(int[] outerSize, int[] innerSize) {
    return new AutoencoderNetwork.Builder(outerSize, innerSize);
  }
  
  private final int[] outerSize;
  private final int[] innerSize;
  private final GaussianNoiseLayer inputNoise;
  private final BiasLayer encoderBias;
  private final DenseSynapseLayer encoderSynapse;
  private final ReLuActivationLayer encoderActivation;
  private final DropoutNoiseLayer encodedNoise;
  private NNLayer decoderSynapse;
  private final BiasLayer decoderBias;
  private final ReLuActivationLayer decoderActivation;
  private final PipelineNetwork encoder;
  private final PipelineNetwork decoder;
  private final AutoencoderNetwork.Builder networkParameters;
  private final VariableLayer decoderSynapsePlaceholder;
  
  protected AutoencoderNetwork(AutoencoderNetwork.Builder networkParameters) {
    this.networkParameters = networkParameters;
    this.outerSize = networkParameters.getOuterSize();
    this.innerSize = networkParameters.getInnerSize();
    
    this.inputNoise = new GaussianNoiseLayer().setValue(networkParameters.getNoise());
    this.encoderSynapse = new DenseSynapseLayer(this.outerSize, this.innerSize);
    this.encoderSynapse.initSpacial(networkParameters.getInitRadius(), networkParameters.getInitStiffness(), networkParameters.getInitPeak());
    this.encoderBias = new BiasLayer(this.innerSize).setWeights(i -> 0.0);
    this.encoderActivation = new ReLuActivationLayer().freeze();
    this.encodedNoise = new DropoutNoiseLayer().setValue(networkParameters.getDropout());
    this.decoderSynapse = new TransposedSynapseLayer(encoderSynapse);
    this.decoderSynapsePlaceholder = new VariableLayer(this.decoderSynapse);
    this.decoderBias = new BiasLayer(this.outerSize).setWeights(i -> 0.0);
    this.decoderActivation = new ReLuActivationLayer().freeze();
    
    this.encoder = new PipelineNetwork();
    this.encoder.add(inputNoise);
    this.encoder.add(encoderSynapse);
    this.encoder.add(encoderBias);
    this.encoder.add(encoderActivation);
    this.encoder.add(encodedNoise);
    
    this.decoder = new PipelineNetwork();
    this.decoder.add(decoderSynapsePlaceholder);
    this.decoder.add(decoderBias);
    this.decoder.add(decoderActivation);
  }
  
  public Tensor[] encode(Tensor[] data) {
    return encoder.getLayer()
               .eval(NNResult.batchResultArray(Arrays.stream(data).map(x -> new Tensor[]{x}).toArray(i -> new Tensor[i][])))
               .data;
  }
  
  public void trainingMode() {
    this.inputNoise.setValue(networkParameters.getNoise());
    this.encodedNoise.setValue(networkParameters.getDropout());
  }
  
  public void runMode() {
    this.inputNoise.setValue(0.0);
    this.encodedNoise.setValue(0.0);
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
  
  public DenseSynapseLayer getEncoderSynapse() {
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
  
    public void run(Tensor... data) {
      SimpleLossNetwork trainingNetwork = getTrainingNetwork();
      StochasticArrayTrainable trainable = new StochasticArrayTrainable(Arrays.stream(data).map(x -> new Tensor[]{x, x}).toArray(i -> new Tensor[i][]), trainingNetwork, getSampleSize());
      L12Normalizer normalized = new ConstL12Normalizer(trainable).setFactor_L1(getL1normalization()).setFactor_L2(getL2normalization());
      IterativeTrainer trainer = new IterativeTrainer(normalized);
      trainer.setOrientation(getOrient());
      trainer.setLineSearchFactory(()->getStep());
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
    
    public AutoencoderNetwork.TrainingParameters setSampleSize(int sampleSize) {
      this.sampleSize = sampleSize;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setL1normalization(double l1normalization) {
      this.l1normalization = l1normalization;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setL2normalization(double l2normalization) {
      this.l2normalization = l2normalization;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setOrient(OrientationStrategy orient) {
      this.orient = orient;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setStep(LineSearchStrategy step) {
      this.step = step;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setMonitor(TrainingMonitor monitor) {
      this.monitor = monitor;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setTimeoutMinutes(int timeoutMinutes) {
      this.timeoutMinutes = timeoutMinutes;
      return this;
    }
    
    public AutoencoderNetwork.TrainingParameters setEndFitness(double endFitness) {
      this.endFitness = endFitness;
      return this;
    }
  
    public AutoencoderNetwork.TrainingParameters setMaxIterations(int maxIterations) {
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
    
    public AutoencoderNetwork.Builder setNoise(double noise) {
      this.noise = noise;
      return this;
    }
    
    public AutoencoderNetwork.Builder setInitRadius(double initRadius) {
      this.initRadius = initRadius;
      return this;
    }
    
    public AutoencoderNetwork.Builder setInitStiffness(int initStiffness) {
      this.initStiffness = initStiffness;
      return this;
    }
    
    public AutoencoderNetwork.Builder setInitPeak(double initPeak) {
      this.initPeak = initPeak;
      return this;
    }
    
    public AutoencoderNetwork.Builder setDropout(double dropout) {
      this.dropout = dropout;
      return this;
    }
    
    public AutoencoderNetwork build() {
      return new AutoencoderNetwork(AutoencoderNetwork.Builder.this);
    }
  }
}
