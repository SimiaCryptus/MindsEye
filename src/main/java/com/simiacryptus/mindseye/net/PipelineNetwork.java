package com.simiacryptus.mindseye.net;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dag.DAGNode;

import java.util.Arrays;
import java.util.HashMap;

public class PipelineNetwork extends DAGNetwork {

    private DAGNode head = getInput().get(0);
    private final HashMap<NNLayer<?>, NNLayer<?>> forwardLinkIndex = new HashMap<>();
    private final HashMap<NNLayer<?>, NNLayer<?>> backwardLinkIndex = new HashMap<>();

    public PipelineNetwork() {
        this(1);
    }

    public PipelineNetwork(int inputs) {
        super(inputs);
    }

    @SafeVarargs
    @Override
    public final DAGNode add(final NNLayer<?> nextHead, final DAGNode... head) {
        DAGNode node = super.add(nextHead, head);
        assert Arrays.stream(head).allMatch(x->x != null);
        if(head.length>0){
            // XXX: Prev/next linking only tracks first input node
            final NNLayer<?> prevHead = getLayer(head[0]);
            this.backwardLinkIndex.put(nextHead, prevHead);
            this.forwardLinkIndex.put(prevHead, nextHead);
        }
        assert null != getInput();
        setHead(node);
        return node;
    }

    public DAGNode add(NNLayer<?> nextHead) {
        DAGNode head = getHead();
        if(null == head) return add(nextHead, getInput(0));
        return add(nextHead, head);
    }

    public DAGNode getHead() {
        return this.head;
    }

    @Override
    public JsonObject getJson() {
        final JsonObject json = super.getJson();
        json.add("root", getHead().toJson());
        return json;
    }

    public void setHead(final DAGNode imageRMS) {
        this.head = imageRMS;
    }

}
