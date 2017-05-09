package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.dag.DAGNode;
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.net.meta.Sparse01MetaLayer;
import com.simiacryptus.mindseye.net.reducers.SumReducerLayer;

public class SparseAutoencoderTrainer extends SupervisedNetwork {

    public final DAGNode encoder;
    public final DAGNode decoder;
    public final DAGNode loss;
    public final DAGNode sparsity;
    public final DAGNode sumSparsityLayer;
    public final DAGNode sumFitnessLayer;
    public final DAGNode sparsityThrottleLayer;

    public SparseAutoencoderTrainer(final NNLayer encoder, final NNLayer decoder) {
        super(1);
        this.encoder = add(encoder, getInput(0));
        this.decoder = add(decoder, this.encoder);
        this.loss = add(new MeanSqLossLayer(), this.decoder, getInput(0));
        this.sparsity = add(new Sparse01MetaLayer(), this.encoder);
        this.sumSparsityLayer = add(new SumReducerLayer(), this.sparsity);
        this.sparsityThrottleLayer = add(new LinearActivationLayer().setWeight(0.5), this.sumSparsityLayer);
        this.sumFitnessLayer = add(new SumReducerLayer(), this.sparsityThrottleLayer, this.loss);
    }

    @Override
    public DAGNode getHead() {
        return this.sumFitnessLayer;
    }
}
