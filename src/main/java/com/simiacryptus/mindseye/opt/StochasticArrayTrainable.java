package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.graph.dag.DAGNetwork;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;

public class StochasticArrayTrainable implements Trainable {

    private final Tensor[][] trainingData;
    private final DAGNetwork network;
    private long hash = Util.R.get().nextLong();
    private int trainingSize = Integer.MAX_VALUE;
    private Tensor[][] sampledData;

    public StochasticArrayTrainable(Tensor[][] trainingData, DAGNetwork network, int trainingSize) {
        this.trainingData = trainingData;
        this.network = network;
        this.trainingSize = trainingSize;
        resetSampling();
    }

    @Override
    public PointSample measure() {
        NNResult result = network.eval(NNResult.batchResultArray(sampledData));
        DeltaSet deltaSet = new DeltaSet();
        result.accumulate(deltaSet);
        DeltaSet stateSet = new DeltaSet();
        deltaSet.map.forEach((layer,layerDelta)->{
            stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target);
        });
        assert(Arrays.stream(result.data).allMatch(x->x.dim()==1));
        double meanValue = Arrays.stream(result.data).mapToDouble(x->x.getData()[0]).average().getAsDouble();
        return new PointSample(deltaSet, stateSet, meanValue);
    }

    @Override
    public void resetToFull() {
        sampledData = trainingData;
    }

    @Override
    public void resetSampling() {
        setHash(Util.R.get().nextLong());
    }

    public int getTrainingSize() {
        return this.trainingSize;
    }

    public StochasticArrayTrainable setTrainingSize(final int trainingSize) {
        this.trainingSize = trainingSize;
        refreshSampledData();
        return this;
    }

    private void setHash(long newValue) {
        if(this.hash == newValue) return;
        this.hash = newValue;
        refreshSampledData();
    }

    private void refreshSampledData() {
        final Tensor[][] rawData = trainingData;
        assert 0 < rawData.length;
        assert 0 < getTrainingSize();
        this.sampledData = java.util.Arrays.stream(rawData).parallel() //
                .sorted(java.util.Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
                .limit(getTrainingSize()) //
                .toArray(i -> new Tensor[i][]);
    }
}
