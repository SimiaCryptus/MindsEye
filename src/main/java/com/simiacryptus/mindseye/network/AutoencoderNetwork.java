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
import com.simiacryptus.mindseye.layers.TensorArray;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.mindseye.layers.activation.DropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.GaussianNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.synapse.TransposedSynapseLayer;
import com.simiacryptus.mindseye.layers.util.VariableLayer;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.trainable.ConstL12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.L12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.util.ml.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Autoencoder network.
 */
public class AutoencoderNetwork {
  
  /**
   * The type Recursive builder.
   */
  public static class RecursiveBuilder {
    
    private final List<TensorList> representations = new ArrayList<>();
    private final List<int[]> dimensions = new ArrayList<>();
    private final List<AutoencoderNetwork> layers = new ArrayList<>();
  
    /**
     * Instantiates a new Recursive builder.
     *
     * @param data the data
     */
    public RecursiveBuilder(TensorList data) {
      representations.add(data);
      dimensions.add(data.get(0).getDimensions());
    }
  
    /**
     * Training mode.
     */
    public void trainingMode() {
      layers.forEach(x->x.trainingMode());
    }
  
    /**
     * Run mode.
     */
    public void runMode() {
      layers.forEach(x->x.runMode());
    }
  
    /**
     * Grow layer autoencoder network.
     *
     * @param dims the dims
     * @return the autoencoder network
     */
    public AutoencoderNetwork growLayer(int... dims) {
      return growLayer(layers.isEmpty()?100:0, 1, 10, dims);
    }
  
    /**
     * Grow layer autoencoder network.
     *
     * @param pretrainingSize    the pretraining size
     * @param pretrainingMinutes the pretraining minutes
     * @param pretrainIterations the pretrain iterations
     * @param dims               the dims
     * @return the autoencoder network
     */
    public AutoencoderNetwork growLayer(int pretrainingSize, int pretrainingMinutes, int pretrainIterations, int[] dims) {
      trainingMode();
      AutoencoderNetwork newLayer = configure(newLayer(dimensions.get(dimensions.size() - 1), dims)).build();

      TensorList data = representations.get(representations.size() - 1);
      dimensions.add(dims);
      layers.add(newLayer);

      if(pretrainingSize > 0 && pretrainIterations > 0 && pretrainingMinutes > 0) {
        ArrayList<Tensor> list = new ArrayList<>(data.stream().collect(Collectors.toList()));
        Collections.shuffle(list);
        Tensor[] pretrainingSet = list.subList(0, pretrainingSize).toArray(new Tensor[]{});
        configure(newLayer.train()).setMaxIterations(pretrainIterations).setTimeoutMinutes(pretrainingMinutes).run(new TensorArray(pretrainingSet));
      }
      newLayer.decoderSynapse = ((TransposedSynapseLayer)newLayer.decoderSynapse).asNewSynapseLayer();
      newLayer.decoderSynapsePlaceholder.setInner(newLayer.decoderSynapse);
      configure(newLayer.train()).run(data);

      runMode();
      representations.add(newLayer.encode(data));
      return newLayer;
    }
  
    /**
     * Tune.
     */
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
  
    /**
     * Configure autoencoder network . training parameters.
     *
     * @param trainingParameters the training parameters
     * @return the autoencoder network . training parameters
     */
    protected AutoencoderNetwork.TrainingParameters configure(AutoencoderNetwork.TrainingParameters trainingParameters) {
      return trainingParameters;
    }
  
    /**
     * Configure autoencoder network . builder.
     *
     * @param builder the builder
     * @return the autoencoder network . builder
     */
    protected AutoencoderNetwork.Builder configure(AutoencoderNetwork.Builder builder) {
      return builder;
    }
  
    /**
     * Echo nn layer.
     *
     * @return the nn layer
     */
    public NNLayer echo() {
      PipelineNetwork network = new PipelineNetwork();
      network.add(getEncoder());
      network.add(getDecoder());
      return network;
    }
  
    /**
     * Gets encoder.
     *
     * @return the encoder
     */
    public NNLayer getEncoder() {
      PipelineNetwork network = new PipelineNetwork();
      for (int i = 0; i < layers.size(); i++) {
        network.add(layers.get(i).getEncoder());
      }
      return network;
    }
  
    /**
     * Gets decoder.
     *
     * @return the decoder
     */
    public NNLayer getDecoder() {
      PipelineNetwork network = new PipelineNetwork();
      for (int i = layers.size() - 1; i >= 0; i--) {
        network.add(layers.get(i).getDecoder());
      }
      return network;
    }
  
    /**
     * Gets layers.
     *
     * @return the layers
     */
    public List<AutoencoderNetwork> getLayers() {
      return Collections.unmodifiableList(layers);
    }
  }
  
  /**
   * Train autoencoder network . training parameters.
   *
   * @return the autoencoder network . training parameters
   */
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
  
  /**
   * New layer autoencoder network . builder.
   *
   * @param outerSize the outer size
   * @param innerSize the inner size
   * @return the autoencoder network . builder
   */
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
  
  /**
   * Instantiates a new Autoencoder network.
   *
   * @param networkParameters the network parameters
   */
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
  
  /**
   * Encode tensor list.
   *
   * @param data the data
   * @return the tensor list
   */
  public TensorList encode(TensorList data) {
    return encoder.getLayer()
               .eval(new NNLayer.NNExecutionContext() {}, NNResult.batchResultArray(data.stream().map(x -> new Tensor[]{x}).toArray(i -> new Tensor[i][])))
               .data;
  }
  
  /**
   * Training mode.
   */
  public void trainingMode() {
    this.inputNoise.setValue(networkParameters.getNoise());
    this.encodedNoise.setValue(networkParameters.getDropout());
  }
  
  /**
   * Run mode.
   */
  public void runMode() {
    this.inputNoise.setValue(0.0);
    this.encodedNoise.setValue(0.0);
  }
  
  /**
   * Get outer size int [ ].
   *
   * @return the int [ ]
   */
  public int[] getOuterSize() {
    return outerSize;
  }
  
  /**
   * Get inner size int [ ].
   *
   * @return the int [ ]
   */
  public int[] getInnerSize() {
    return innerSize;
  }
  
  /**
   * Gets input noise.
   *
   * @return the input noise
   */
  public GaussianNoiseLayer getInputNoise() {
    return inputNoise;
  }
  
  /**
   * Gets encoder bias.
   *
   * @return the encoder bias
   */
  public BiasLayer getEncoderBias() {
    return encoderBias;
  }
  
  /**
   * Gets encoder synapse.
   *
   * @return the encoder synapse
   */
  public DenseSynapseLayer getEncoderSynapse() {
    return encoderSynapse;
  }
  
  /**
   * Gets encoder activation.
   *
   * @return the encoder activation
   */
  public NNLayer getEncoderActivation() {
    return encoderActivation;
  }
  
  /**
   * Gets encoded noise.
   *
   * @return the encoded noise
   */
  public DropoutNoiseLayer getEncodedNoise() {
    return encodedNoise;
  }
  
  /**
   * Gets decoder synapse.
   *
   * @return the decoder synapse
   */
  public NNLayer getDecoderSynapse() {
    return decoderSynapse;
  }
  
  /**
   * Gets decoder bias.
   *
   * @return the decoder bias
   */
  public BiasLayer getDecoderBias() {
    return decoderBias;
  }
  
  /**
   * Gets decoder activation.
   *
   * @return the decoder activation
   */
  public NNLayer getDecoderActivation() {
    return decoderActivation;
  }
  
  /**
   * Gets encoder.
   *
   * @return the encoder
   */
  public NNLayer getEncoder() {
    return encoder;
  }
  
  /**
   * Gets decoder.
   *
   * @return the decoder
   */
  public NNLayer getDecoder() {
    return decoder;
  }
  
  /**
   * The type Training parameters.
   */
  public static abstract class TrainingParameters {
    private int sampleSize = Integer.MAX_VALUE;
    private double l1normalization = 0.0;
    private double l2normalization = 0.0;
    private OrientationStrategy orient = new LBFGS().setMinHistory(5).setMaxHistory(35);
    private LineSearchStrategy step = new ArmijoWolfeSearch().setC2(0.9).setAlpha(1e-4);
    private TrainingMonitor monitor = null;
    private int timeoutMinutes = 10;
    private double endFitness = Double.NEGATIVE_INFINITY;
    private int maxIterations = Integer.MAX_VALUE;
  
    /**
     * Run.
     *
     * @param data the data
     */
    public void run(TensorList data) {
      SimpleLossNetwork trainingNetwork = getTrainingNetwork();
      StochasticArrayTrainable trainable = new StochasticArrayTrainable(data.stream().map(x -> new Tensor[]{x, x}).toArray(i -> new Tensor[i][]), trainingNetwork, getSampleSize());
      L12Normalizer normalized = new ConstL12Normalizer(trainable).setFactor_L1(getL1normalization()).setFactor_L2(getL2normalization());
      IterativeTrainer trainer = new IterativeTrainer(normalized);
      trainer.setOrientation(getOrient());
      trainer.setLineSearchFactory((s)->getStep());
      TrainingMonitor monitor = getMonitor();
      trainer.setMonitor(wrap(monitor));
      trainer.setTimeout(getTimeoutMinutes(), TimeUnit.MINUTES);
      trainer.setTerminateThreshold(getEndFitness());
      trainer.setMaxIterations(maxIterations);
      trainer.run();
    }
  
    /**
     * Wrap training monitor.
     *
     * @param monitor the monitor
     * @return the training monitor
     */
    protected abstract TrainingMonitor wrap(TrainingMonitor monitor);
  
    /**
     * Gets training network.
     *
     * @return the training network
     */
    public abstract SimpleLossNetwork getTrainingNetwork();
  
    /**
     * Gets sample size.
     *
     * @return the sample size
     */
    public int getSampleSize() {
      return sampleSize;
    }
  
    /**
     * Gets l 1 normalization.
     *
     * @return the l 1 normalization
     */
    public double getL1normalization() {
      return l1normalization;
    }
  
    /**
     * Gets l 2 normalization.
     *
     * @return the l 2 normalization
     */
    public double getL2normalization() {
      return l2normalization;
    }
  
    /**
     * Gets orient.
     *
     * @return the orient
     */
    public OrientationStrategy getOrient() {
      return orient;
    }
  
    /**
     * Gets step.
     *
     * @return the step
     */
    public LineSearchStrategy getStep() {
      return step;
    }
  
    /**
     * Gets monitor.
     *
     * @return the monitor
     */
    public TrainingMonitor getMonitor() {
      return monitor;
    }
  
    /**
     * Gets timeout minutes.
     *
     * @return the timeout minutes
     */
    public int getTimeoutMinutes() {
      return timeoutMinutes;
    }
  
    /**
     * Gets end fitness.
     *
     * @return the end fitness
     */
    public double getEndFitness() {
      return endFitness;
    }
  
    /**
     * Sets sample size.
     *
     * @param sampleSize the sample size
     * @return the sample size
     */
    public AutoencoderNetwork.TrainingParameters setSampleSize(int sampleSize) {
      this.sampleSize = sampleSize;
      return this;
    }
  
    /**
     * Sets l 1 normalization.
     *
     * @param l1normalization the l 1 normalization
     * @return the l 1 normalization
     */
    public AutoencoderNetwork.TrainingParameters setL1normalization(double l1normalization) {
      this.l1normalization = l1normalization;
      return this;
    }
  
    /**
     * Sets l 2 normalization.
     *
     * @param l2normalization the l 2 normalization
     * @return the l 2 normalization
     */
    public AutoencoderNetwork.TrainingParameters setL2normalization(double l2normalization) {
      this.l2normalization = l2normalization;
      return this;
    }
  
    /**
     * Sets orient.
     *
     * @param orient the orient
     * @return the orient
     */
    public AutoencoderNetwork.TrainingParameters setOrient(OrientationStrategy orient) {
      this.orient = orient;
      return this;
    }
  
    /**
     * Sets step.
     *
     * @param step the step
     * @return the step
     */
    public AutoencoderNetwork.TrainingParameters setStep(LineSearchStrategy step) {
      this.step = step;
      return this;
    }
  
    /**
     * Sets monitor.
     *
     * @param monitor the monitor
     * @return the monitor
     */
    public AutoencoderNetwork.TrainingParameters setMonitor(TrainingMonitor monitor) {
      this.monitor = monitor;
      return this;
    }
  
    /**
     * Sets timeout minutes.
     *
     * @param timeoutMinutes the timeout minutes
     * @return the timeout minutes
     */
    public AutoencoderNetwork.TrainingParameters setTimeoutMinutes(int timeoutMinutes) {
      this.timeoutMinutes = timeoutMinutes;
      return this;
    }
  
    /**
     * Sets end fitness.
     *
     * @param endFitness the end fitness
     * @return the end fitness
     */
    public AutoencoderNetwork.TrainingParameters setEndFitness(double endFitness) {
      this.endFitness = endFitness;
      return this;
    }
  
    /**
     * Sets max iterations.
     *
     * @param maxIterations the max iterations
     * @return the max iterations
     */
    public AutoencoderNetwork.TrainingParameters setMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
    }
  
    /**
     * Gets max iterations.
     *
     * @return the max iterations
     */
    public int getMaxIterations() {
      return maxIterations;
    }
  }
  
  /**
   * The type Builder.
   */
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
  
    /**
     * Get outer size int [ ].
     *
     * @return the int [ ]
     */
    public int[] getOuterSize() {
      return outerSize;
    }
  
    /**
     * Get inner size int [ ].
     *
     * @return the int [ ]
     */
    public int[] getInnerSize() {
      return innerSize;
    }
  
    /**
     * Gets noise.
     *
     * @return the noise
     */
    public double getNoise() {
      return noise;
    }
  
    /**
     * Gets init radius.
     *
     * @return the init radius
     */
    public double getInitRadius() {
      return initRadius;
    }
  
    /**
     * Gets init stiffness.
     *
     * @return the init stiffness
     */
    public int getInitStiffness() {
      return initStiffness;
    }
  
    /**
     * Gets init peak.
     *
     * @return the init peak
     */
    public double getInitPeak() {
      return initPeak;
    }
  
    /**
     * Gets dropout.
     *
     * @return the dropout
     */
    public double getDropout() {
      return dropout;
    }
  
    /**
     * Sets noise.
     *
     * @param noise the noise
     * @return the noise
     */
    public AutoencoderNetwork.Builder setNoise(double noise) {
      this.noise = noise;
      return this;
    }
  
    /**
     * Sets init radius.
     *
     * @param initRadius the init radius
     * @return the init radius
     */
    public AutoencoderNetwork.Builder setInitRadius(double initRadius) {
      this.initRadius = initRadius;
      return this;
    }
  
    /**
     * Sets init stiffness.
     *
     * @param initStiffness the init stiffness
     * @return the init stiffness
     */
    public AutoencoderNetwork.Builder setInitStiffness(int initStiffness) {
      this.initStiffness = initStiffness;
      return this;
    }
  
    /**
     * Sets init peak.
     *
     * @param initPeak the init peak
     * @return the init peak
     */
    public AutoencoderNetwork.Builder setInitPeak(double initPeak) {
      this.initPeak = initPeak;
      return this;
    }
  
    /**
     * Sets dropout.
     *
     * @param dropout the dropout
     * @return the dropout
     */
    public AutoencoderNetwork.Builder setDropout(double dropout) {
      this.dropout = dropout;
      return this;
    }
  
    /**
     * Build autoencoder network.
     *
     * @return the autoencoder network
     */
    public AutoencoderNetwork build() {
      return new AutoencoderNetwork(AutoencoderNetwork.Builder.this);
    }
  }
}
