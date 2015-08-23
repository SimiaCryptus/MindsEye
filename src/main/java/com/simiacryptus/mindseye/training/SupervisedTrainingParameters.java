package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SupervisedTrainingParameters {
  private PipelineNetwork net;
  private final NDArray[][] trainingData;
  private double weight = 1;
  
  protected SupervisedTrainingParameters() {
    super();
    this.net = null;
    this.trainingData = null;
  }

  public SupervisedTrainingParameters(final PipelineNetwork net, final NDArray[][] trainingData) {
    this.net = net;
    this.trainingData = trainingData;
  }
  
  public NDArray getIdeal(final NNResult eval, final NDArray preset) {
    return preset;
  }
  
  public PipelineNetwork getNet() {
    return this.net;
  }
  
  public final NDArray[][] getTrainingData() {
    return this.trainingData;
  }

  public double getWeight() {
    return this.weight;
  }

  public void setNet(final PipelineNetwork net) {
    this.net = net;
  }

  public SupervisedTrainingParameters setWeight(final double weight) {
    this.weight = weight;
    return this;
  }
}