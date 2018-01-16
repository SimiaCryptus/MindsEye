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

package com.simiacryptus.mindseye.network.util;

import com.simiacryptus.mindseye.eval.ConstL12Normalizer;
import com.simiacryptus.mindseye.eval.L12Normalizer;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Autoencoder network.
 */
public class AutoencoderNetwork {
  
  private final PipelineNetwork decoder;
  private final ReLuActivationLayer decoderActivation;
  private final BiasLayer decoderBias;
  private final VariableLayer decoderSynapsePlaceholder;
  private final DropoutNoiseLayer encodedNoise;
  private final PipelineNetwork encoder;
  private final ReLuActivationLayer encoderActivation;
  private final BiasLayer encoderBias;
  private final FullyConnectedLayer encoderSynapse;
  private final int[] innerSize;
  private final GaussianNoiseLayer inputNoise;
  private final AutoencoderNetwork.Builder networkParameters;
  private final int[] outerSize;
  private NNLayer decoderSynapse;
  
  /**
   * Instantiates a new Autoencoder network.
   *
   * @param networkParameters the network parameters
   */
  protected AutoencoderNetwork(final AutoencoderNetwork.Builder networkParameters) {
    this.networkParameters = networkParameters;
    outerSize = networkParameters.getOuterSize();
    innerSize = networkParameters.getInnerSize();
  
    inputNoise = new GaussianNoiseLayer().setValue(networkParameters.getNoise());
    encoderSynapse = new FullyConnectedLayer(outerSize, innerSize);
    encoderSynapse.initSpacial(networkParameters.getInitRadius(), networkParameters.getInitStiffness(), networkParameters.getInitPeak());
    encoderBias = new BiasLayer(innerSize).setWeights(i -> 0.0);
    encoderActivation = (ReLuActivationLayer) new ReLuActivationLayer().freeze();
    encodedNoise = new DropoutNoiseLayer().setValue(networkParameters.getDropout());
    decoderSynapse = encoderSynapse.getTranspose();
    decoderSynapsePlaceholder = new VariableLayer(decoderSynapse);
    decoderBias = new BiasLayer(outerSize).setWeights(i -> 0.0);
    decoderActivation = (ReLuActivationLayer) new ReLuActivationLayer().freeze();
  
    encoder = new PipelineNetwork();
    encoder.add(inputNoise);
    encoder.add(encoderSynapse);
    encoder.add(encoderBias);
    encoder.add(encoderActivation);
    encoder.add(encodedNoise);
  
    decoder = new PipelineNetwork();
    decoder.add(decoderSynapsePlaceholder);
    decoder.add(decoderBias);
    decoder.add(decoderActivation);
  }
  
  /**
   * New layer autoencoder network . builder.
   *
   * @param outerSize the outer size
   * @param innerSize the localCopy size
   * @return the autoencoder network . builder
   */
  public static AutoencoderNetwork.Builder newLayer(final int[] outerSize, final int[] innerSize) {
    return new AutoencoderNetwork.Builder(outerSize, innerSize);
  }
  
  /**
   * Encode tensor list.
   *
   * @param data the data
   * @return the tensor list
   */
  public TensorList encode(final TensorList data) {
    return encoder.getLayer()
                  .eval(NNConstant.batchResultArray(data.stream().map(x -> new Tensor[]{x}).toArray(i -> new Tensor[i][])))
                  .getData();
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
   * Gets decoder activation.
   *
   * @return the decoder activation
   */
  public NNLayer getDecoderActivation() {
    return decoderActivation;
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
   * Gets decoder synapse.
   *
   * @return the decoder synapse
   */
  public NNLayer getDecoderSynapse() {
    return decoderSynapse;
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
   * Gets encoder.
   *
   * @return the encoder
   */
  public NNLayer getEncoder() {
    return encoder;
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
  public FullyConnectedLayer getEncoderSynapse() {
    return encoderSynapse;
  }
  
  /**
   * Get localCopy size int [ ].
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
   * Get outer size int [ ].
   *
   * @return the int [ ]
   */
  public int[] getOuterSize() {
    return outerSize;
  }
  
  /**
   * Run mode.
   */
  public void runMode() {
    inputNoise.setValue(0.0);
    encodedNoise.setValue(0.0);
  }
  
  /**
   * Train autoencoder network . training parameters.
   *
   * @return the autoencoder network . training parameters
   */
  public AutoencoderNetwork.TrainingParameters train() {
    return new AutoencoderNetwork.TrainingParameters() {
      @Override
      public SimpleLossNetwork getTrainingNetwork() {
        final PipelineNetwork student = new PipelineNetwork();
        student.add(encoder);
        student.add(decoder);
        return new SimpleLossNetwork(student, new MeanSqLossLayer());
      }
  
      @Override
      protected TrainingMonitor wrap(final TrainingMonitor monitor) {
        return new TrainingMonitor() {
          @Override
          public void log(final String msg) {
            monitor.log(msg);
          }
  
          @Override
          public void onStepComplete(final Step currentPoint) {
            inputNoise.shuffle();
            encodedNoise.shuffle();
            monitor.onStepComplete(currentPoint);
          }
        };
      }
    };
  }
  
  /**
   * Training mode.
   */
  public void trainingMode() {
    inputNoise.setValue(networkParameters.getNoise());
    encodedNoise.setValue(networkParameters.getDropout());
  }
  
  /**
   * The type Builder.
   */
  public static class Builder {
  
    private final int[] innerSize;
    private final int[] outerSize;
    private double dropout = 0.0;
    private double initPeak = 0.001;
    private double initRadius = 0.5;
    private int initStiffness = 3;
    private double noise = 0.0;
  
    private Builder(final int[] outerSize, final int[] innerSize) {
      this.outerSize = outerSize;
      this.innerSize = innerSize;
    }
  
    /**
     * Build autoencoder network.
     *
     * @return the autoencoder network
     */
    public AutoencoderNetwork build() {
      return new AutoencoderNetwork(AutoencoderNetwork.Builder.this);
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
     * Sets dropout.
     *
     * @param dropout the dropout
     * @return the dropout
     */
    public AutoencoderNetwork.Builder setDropout(final double dropout) {
      this.dropout = dropout;
      return this;
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
     * Sets init peak.
     *
     * @param initPeak the init peak
     * @return the init peak
     */
    public AutoencoderNetwork.Builder setInitPeak(final double initPeak) {
      this.initPeak = initPeak;
      return this;
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
     * Sets init radius.
     *
     * @param initRadius the init radius
     * @return the init radius
     */
    public AutoencoderNetwork.Builder setInitRadius(final double initRadius) {
      this.initRadius = initRadius;
      return this;
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
     * Sets init stiffness.
     *
     * @param initStiffness the init stiffness
     * @return the init stiffness
     */
    public AutoencoderNetwork.Builder setInitStiffness(final int initStiffness) {
      this.initStiffness = initStiffness;
      return this;
    }
  
    /**
     * Get localCopy size int [ ].
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
     * Sets noise.
     *
     * @param noise the noise
     * @return the noise
     */
    public AutoencoderNetwork.Builder setNoise(final double noise) {
      this.noise = noise;
      return this;
    }
  
    /**
     * Get outer size int [ ].
     *
     * @return the int [ ]
     */
    public int[] getOuterSize() {
      return outerSize;
    }
  }
  
  /**
   * The type Recursive builder.
   */
  public static class RecursiveBuilder {
  
    private final List<int[]> dimensions = new ArrayList<>();
    private final List<AutoencoderNetwork> layers = new ArrayList<>();
    private final List<TensorList> representations = new ArrayList<>();
  
    /**
     * Instantiates a new Recursive builder.
     *
     * @param data the data
     */
    public RecursiveBuilder(final TensorList data) {
      representations.add(data);
      dimensions.add(data.get(0).getDimensions());
    }
  
    /**
     * Configure autoencoder network . builder.
     *
     * @param builder the builder
     * @return the autoencoder network . builder
     */
    protected AutoencoderNetwork.Builder configure(final AutoencoderNetwork.Builder builder) {
      return builder;
    }
  
    /**
     * Configure autoencoder network . training parameters.
     *
     * @param trainingParameters the training parameters
     * @return the autoencoder network . training parameters
     */
    protected AutoencoderNetwork.TrainingParameters configure(final AutoencoderNetwork.TrainingParameters trainingParameters) {
      return trainingParameters;
    }
  
    /**
     * Echo nn layer.
     *
     * @return the nn layer
     */
    public NNLayer echo() {
      final PipelineNetwork network = new PipelineNetwork();
      network.add(getEncoder());
      network.add(getDecoder());
      return network;
    }
  
    /**
     * Gets decoder.
     *
     * @return the decoder
     */
    public NNLayer getDecoder() {
      final PipelineNetwork network = new PipelineNetwork();
      for (int i = layers.size() - 1; i >= 0; i--) {
        network.add(layers.get(i).getDecoder());
      }
      return network;
    }
  
    /**
     * Gets encoder.
     *
     * @return the encoder
     */
    public NNLayer getEncoder() {
      final PipelineNetwork network = new PipelineNetwork();
      for (int i = 0; i < layers.size(); i++) {
        network.add(layers.get(i).getEncoder());
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
  
    /**
     * Grow layer autoencoder network.
     *
     * @param dims the dims
     * @return the autoencoder network
     */
    public AutoencoderNetwork growLayer(final int... dims) {
      return growLayer(layers.isEmpty() ? 100 : 0, 1, 10, dims);
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
    public AutoencoderNetwork growLayer(final int pretrainingSize, final int pretrainingMinutes, final int pretrainIterations, final int[] dims) {
      trainingMode();
      final AutoencoderNetwork newLayer = configure(AutoencoderNetwork.newLayer(dimensions.get(dimensions.size() - 1), dims)).build();
  
      final TensorList data = representations.get(representations.size() - 1);
      dimensions.add(dims);
      layers.add(newLayer);
  
      if (pretrainingSize > 0 && pretrainIterations > 0 && pretrainingMinutes > 0) {
        final ArrayList<Tensor> list = new ArrayList<>(data.stream().collect(Collectors.toList()));
        Collections.shuffle(list);
        final Tensor[] pretrainingSet = list.subList(0, pretrainingSize).toArray(new Tensor[]{});
        configure(newLayer.train()).setMaxIterations(pretrainIterations).setTimeoutMinutes(pretrainingMinutes).run(new TensorArray(pretrainingSet));
      }
      newLayer.decoderSynapse = ((FullyConnectedLayer) newLayer.decoderSynapse).getTranspose();
      newLayer.decoderSynapsePlaceholder.setInner(newLayer.decoderSynapse);
      configure(newLayer.train()).run(data);
  
      runMode();
      representations.add(newLayer.encode(data));
      return newLayer;
    }
  
    /**
     * Run mode.
     */
    public void runMode() {
      layers.forEach(x -> x.runMode());
    }
  
    /**
     * Training mode.
     */
    public void trainingMode() {
      layers.forEach(x -> x.trainingMode());
    }
  
    /**
     * Tune.
     */
    public void tune() {
      configure(new AutoencoderNetwork.TrainingParameters() {
        @Override
        public SimpleLossNetwork getTrainingNetwork() {
          final PipelineNetwork student = new PipelineNetwork();
          student.add(getEncoder());
          student.add(getDecoder());
          return new SimpleLossNetwork(student, new MeanSqLossLayer());
        }
  
        @Override
        protected TrainingMonitor wrap(final TrainingMonitor monitor) {
          return new TrainingMonitor() {
            @Override
            public void log(final String msg) {
              monitor.log(msg);
            }
  
            @Override
            public void onStepComplete(final Step currentPoint) {
              layers.forEach(layer -> {
                layer.inputNoise.shuffle();
                layer.encodedNoise.shuffle();
              });
              monitor.onStepComplete(currentPoint);
            }
          };
        }
      }).run(representations.get(0));
    }
  }
  
  /**
   * The type Training parameters.
   */
  public abstract static class TrainingParameters {
    private double endFitness = Double.NEGATIVE_INFINITY;
    private double l1normalization = 0.0;
    private double l2normalization = 0.0;
    private int maxIterations = Integer.MAX_VALUE;
    private TrainingMonitor monitor = null;
    private OrientationStrategy<?> orient = new LBFGS().setMinHistory(5).setMaxHistory(35);
    private int sampleSize = Integer.MAX_VALUE;
    private LineSearchStrategy step = new ArmijoWolfeSearch().setC2(0.9).setAlpha(1e-4);
    private int timeoutMinutes = 10;
  
    /**
     * Gets end fitness.
     *
     * @return the end fitness
     */
    public double getEndFitness() {
      return endFitness;
    }
  
    /**
     * Sets end fitness.
     *
     * @param endFitness the end fitness
     * @return the end fitness
     */
    public AutoencoderNetwork.TrainingParameters setEndFitness(final double endFitness) {
      this.endFitness = endFitness;
      return this;
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
     * Sets l 1 normalization.
     *
     * @param l1normalization the l 1 normalization
     * @return the l 1 normalization
     */
    public AutoencoderNetwork.TrainingParameters setL1normalization(final double l1normalization) {
      this.l1normalization = l1normalization;
      return this;
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
     * Sets l 2 normalization.
     *
     * @param l2normalization the l 2 normalization
     * @return the l 2 normalization
     */
    public AutoencoderNetwork.TrainingParameters setL2normalization(final double l2normalization) {
      this.l2normalization = l2normalization;
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
  
    /**
     * Sets max iterations.
     *
     * @param maxIterations the max iterations
     * @return the max iterations
     */
    public AutoencoderNetwork.TrainingParameters setMaxIterations(final int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
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
     * Sets monitor.
     *
     * @param monitor the monitor
     * @return the monitor
     */
    public AutoencoderNetwork.TrainingParameters setMonitor(final TrainingMonitor monitor) {
      this.monitor = monitor;
      return this;
    }
  
    /**
     * Gets orient.
     *
     * @return the orient
     */
    public OrientationStrategy<?> getOrient() {
      return orient;
    }
  
    /**
     * Sets orient.
     *
     * @param orient the orient
     * @return the orient
     */
    public AutoencoderNetwork.TrainingParameters setOrient(final OrientationStrategy<?> orient) {
      this.orient = orient;
      return this;
    }
  
    /**
     * Gets sample size.
     *
     * @return the sample size
     */
    public int getSampleSize() {
      return sampleSize;
    }
  
    /**
     * Sets sample size.
     *
     * @param sampleSize the sample size
     * @return the sample size
     */
    public AutoencoderNetwork.TrainingParameters setSampleSize(final int sampleSize) {
      this.sampleSize = sampleSize;
      return this;
    }
  
    /**
     * Gets runStep.
     *
     * @return the runStep
     */
    public LineSearchStrategy getStep() {
      return step;
    }
  
    /**
     * Sets runStep.
     *
     * @param step the runStep
     * @return the runStep
     */
    public AutoencoderNetwork.TrainingParameters setStep(final LineSearchStrategy step) {
      this.step = step;
      return this;
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
     * Sets timeout minutes.
     *
     * @param timeoutMinutes the timeout minutes
     * @return the timeout minutes
     */
    public AutoencoderNetwork.TrainingParameters setTimeoutMinutes(final int timeoutMinutes) {
      this.timeoutMinutes = timeoutMinutes;
      return this;
    }
  
    /**
     * Gets training network.
     *
     * @return the training network
     */
    public abstract SimpleLossNetwork getTrainingNetwork();
  
    /**
     * Run.
     *
     * @param data the data
     */
    public void run(final TensorList data) {
      final SimpleLossNetwork trainingNetwork = getTrainingNetwork();
      final Trainable trainable = new SampledArrayTrainable(data.stream().map(x -> new Tensor[]{x, x}).toArray(i -> new Tensor[i][]), trainingNetwork, getSampleSize());
      final L12Normalizer normalized = new ConstL12Normalizer(trainable).setFactor_L1(getL1normalization()).setFactor_L2(getL2normalization());
      final IterativeTrainer trainer = new IterativeTrainer(normalized);
      trainer.setOrientation(getOrient());
      trainer.setLineSearchFactory((s) -> getStep());
      final TrainingMonitor monitor = getMonitor();
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
  }
}
