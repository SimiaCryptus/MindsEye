package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.net.dag.DAGNode;

public class SimpleLossNetwork extends SupervisedNetwork {


    private final DAGNode studentNode;
    private final DAGNode lossNode;

    public SimpleLossNetwork(final NNLayer<?> student, final NNLayer<?> loss) {
        super(2);
        studentNode = add(student, getInput(0));
        lossNode = add(loss, studentNode, getInput(1));
    }

    @Override
    public DAGNode getHead() {
        return lossNode;
    }
}
