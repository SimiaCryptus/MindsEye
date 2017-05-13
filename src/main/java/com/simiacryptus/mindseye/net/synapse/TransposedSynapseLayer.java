package com.simiacryptus.mindseye.net.synapse;

import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;

public class TransposedSynapseLayer extends MappedSynapseLayer {

  private final DenseSynapseLayer sibling;

  protected TransposedSynapseLayer() {
    super();
    sibling = null;
  }

  public TransposedSynapseLayer(final DenseSynapseLayer sibling) {
    super(sibling.outputDims, sibling.inputDims);
    this.sibling = sibling;
  }

  @Override
  public int getMappedIndex(Coordinate inputCoord, Coordinate outputCoord) {
    return inputCoord.index + Tensor.dim(inputDims) * outputCoord.index;
  }

  @Override
  public Tensor buildWeights() {
    Tensor weights = sibling.getWeights();
    int[] dims = weights.getDims();
    assert(2 == dims.length);
    double[] data = weights.getData();
    return new Tensor(new int[]{dims[1],dims[0]}, data);
  }

}
