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

package com.simiacryptus.mindseye.network;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.layers.java.WrapperLayer;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Directed Acyclical Graph Network The base class for all conventional network wiring.
 */
@SuppressWarnings("serial")
public abstract class DAGNetwork extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);
  /**
   * The Input handles.
   */
  public final List<UUID> inputHandles;
  /**
   * The Input nodes.
   */
  public final LinkedHashMap<UUID, InputNode> inputNodes;
  /**
   * The Labels.
   */
  protected final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
  /**
   * The Layers by id.
   */
  protected final LinkedHashMap<Object, NNLayer> layersById = new LinkedHashMap<>();
  /**
   * The Nodes by id.
   */
  protected final LinkedHashMap<UUID, DAGNode> nodesById = new LinkedHashMap<>();
  
  /**
   * Instantiates a new Dag network.
   *
   * @param inputs the inputs
   */
  public DAGNetwork(final int inputs) {
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for (int i = 0; i < inputs; i++) {
      addInput();
    }
  }
  
  /**
   * Instantiates a new Dag network.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected DAGNetwork(final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for (final JsonElement item : json.getAsJsonArray("inputs")) {
      final UUID key = UUID.fromString(item.getAsString());
      inputHandles.add(key);
      inputNodes.put(key, new InputNode(this, key));
    }
    final JsonObject jsonNodes = json.getAsJsonObject("nodes");
    final JsonObject jsonLayers = json.getAsJsonObject("layers");
    final JsonObject jsonLinks = json.getAsJsonObject("links");
    final JsonObject jsonLabels = json.getAsJsonObject("labels");
    final Map<UUID, NNLayer> source_layersByNodeId = new HashMap<>();
    final Map<UUID, NNLayer> source_layersByLayerId = new HashMap<>();
    for (final Entry<String, JsonElement> e : jsonLayers.entrySet()) {
      source_layersByLayerId.put(UUID.fromString(e.getKey()), NNLayer.fromJson(e.getValue().getAsJsonObject(), rs));
    }
    for (final Entry<String, JsonElement> e : jsonNodes.entrySet()) {
      final UUID nodeId = UUID.fromString(e.getKey());
      final UUID layerId = UUID.fromString(e.getValue().getAsString());
      final NNLayer layer = source_layersByLayerId.get(layerId);
      assert null != layer;
      source_layersByNodeId.put(nodeId, layer);
    }
    final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
    for (final Entry<String, JsonElement> e : jsonLabels.entrySet()) {
      labels.put(e.getKey(), UUID.fromString(e.getValue().getAsString()));
    }
    final Map<UUID, List<UUID>> deserializedLinks = new HashMap<>();
    for (final Entry<String, JsonElement> e : jsonLinks.entrySet()) {
      final ArrayList<UUID> linkList = new ArrayList<>();
      for (final JsonElement linkItem : e.getValue().getAsJsonArray()) {
        linkList.add(UUID.fromString(linkItem.getAsString()));
      }
      deserializedLinks.put(UUID.fromString(e.getKey()), linkList);
    }
    for (final UUID key : labels.values()) {
      initLinks(deserializedLinks, source_layersByNodeId, key);
    }
    final UUID head = UUID.fromString(json.getAsJsonPrimitive("head").getAsString());
    initLinks(deserializedLinks, source_layersByNodeId, head);
    this.labels.putAll(labels);
    assertConsistent();
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  public DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    return add(null, nextHead, head);
  }
  
  /**
   * Add dag node.
   *
   * @param label the label
   * @param layer the layer
   * @param head  the head
   * @return the dag node
   */
  public DAGNode add(final String label, final NNLayer layer, final DAGNode... head) {
    assertConsistent();
    assert null != getInput();
    final InnerNode node = new InnerNode(this, layer, head);
    layersById.put(layer.getId(), layer);
    nodesById.put(node.getId(), node);
    if (null != label) {
      labels.put(label, node.getId());
    }
    assertConsistent();
    return node;
  }
  
  /**
   * Add input nn layer.
   *
   * @return the nn layer
   */
  public NNLayer addInput() {
    final UUID key = UUID.randomUUID();
    inputHandles.add(key);
    inputNodes.put(key, new InputNode(this, key));
    return this;
  }
  
  /**
   * Assert consistent boolean.
   *
   * @return the boolean
   */
  protected boolean assertConsistent() {
    assert null != getInput();
    for (final Entry<String, UUID> e : labels.entrySet()) {
      assert nodesById.containsKey(e.getValue());
    }
    for (final Entry<UUID, DAGNode> e : nodesById.entrySet()) {
      final NNLayer layer = e.getValue().getLayer();
      assert layersById.containsKey(layer.getId());
      assert layersById.get(layer.getId()) == layer;
    }
    return true;
  }
  
  /**
   * Attach.
   *
   * @param obj the obj
   */
  public void attach(final MonitoredObject obj) {
    visitLayers(layer -> {
      if (layer instanceof MonitoredItem) {
        obj.addObj(layer.getName(), (MonitoredItem) layer);
      }
    });
  }
  
  /**
   * Build exe ctx graph evaluation context.
   *
   * @param inputs the inputs
   * @return the graph evaluation context
   */
  public GraphEvaluationContext buildExeCtx(final NNResult... inputs) {
    assert inputs.length == inputHandles.size() : inputs.length + " != " + inputHandles.size();
    final GraphEvaluationContext graphEvaluationContext = new GraphEvaluationContext();
    for (int i = 0; i < inputs.length; i++) {
      graphEvaluationContext.cache.put(inputHandles.get(i), new CountingNNResult(inputs[i]));
    }
    return graphEvaluationContext;
  }
  
  @Override
  public DAGNetwork copy() {
    return (DAGNetwork) super.copy();
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... input) {
    final GraphEvaluationContext exeCtx = buildExeCtx(input);
    final NNResult result = get(nncontext, exeCtx);
    //exeCtx.finalize();
    return result;
  }
  
  /**
   * Get nn result.
   *
   * @param nncontext   the nncontext
   * @param buildExeCtx the build exe ctx
   * @return the nn result
   */
  public NNResult get(final NNExecutionContext nncontext, final GraphEvaluationContext buildExeCtx) {
    return getHead().get(nncontext, buildExeCtx);
  }
  
  /**
   * Gets by label.
   *
   * @param key the key
   * @return the by label
   */
  public DAGNode getByLabel(final String key) {
    return nodesById.get(labels.get(key));
  }
  
  /**
   * Gets by name.
   *
   * @param <T>  the type parameter
   * @param name the name
   * @return the by name
   */
  @SuppressWarnings("unchecked")
  public <T extends NNLayer> T getByName(final String name) {
    if (null == name) return null;
    final AtomicReference<NNLayer> result = new AtomicReference<>();
    visitLayers(n -> {
      if (name.equals(n.getName())) {
        result.set(n);
      }
    });
    return (T) result.get();
  }
  
  /**
   * Gets child node.
   *
   * @param id the id
   * @return the child node
   */
  public DAGNode getChildNode(final UUID id) {
    if (nodesById.containsKey(id)) {
      return nodesById.get(id);
    }
    return nodesById.values().stream().map(x -> x.getLayer())
                    .filter(x -> x instanceof DAGNetwork)
                    .map(x -> ((DAGNetwork) x).getChildNode(id)).findAny().orElse(null);
  }
  
  @Override
  public List<NNLayer> getChildren() {
    return layersById.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId().toString())).collect(Collectors.toList());
  }
  
  private DAGNode[] getDependencies(final Map<UUID, List<UUID>> deserializedLinks, final UUID e) {
    final List<UUID> links = deserializedLinks.get(e);
    if (null == links) return new DAGNode[]{};
    return links.stream().map(id -> getNode(id)).toArray(i -> new DAGNode[i]);
  }
  
  /**
   * Gets head.
   *
   * @return the head
   */
  public abstract DAGNode getHead();
  
  /**
   * Gets input.
   *
   * @return the input
   */
  public List<DAGNode> getInput() {
    final ArrayList<DAGNode> list = new ArrayList<>();
    for (final UUID key : inputHandles) {
      list.add(inputNodes.get(key));
    }
    return list;
  }
  
  /**
   * Gets input.
   *
   * @param index the index
   * @return the input
   */
  public DAGNode getInput(final int index) {
    final DAGNode input = inputNodes.get(inputHandles.get(index));
    assert null != input;
    return input;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    final JsonArray inputs = new JsonArray();
    json.add("inputs", inputs);
    inputHandles.forEach(uuid -> inputs.add(new JsonPrimitive(uuid.toString())));
    final JsonObject layerMap = new JsonObject();
    final JsonObject nodeMap = new JsonObject();
    final JsonObject links = new JsonObject();
    nodesById.values().forEach(node -> {
      final JsonArray linkArray = new JsonArray();
      Arrays.stream(node.getInputs()).forEach((final DAGNode input) -> linkArray.add(new JsonPrimitive(input.getId().toString())));
      final NNLayer layer = node.getLayer();
      final String nodeId = node.getId().toString();
      final String layerId = layer.getId().toString();
      nodeMap.addProperty(nodeId, layerId);
      layerMap.add(layerId, layer.getJson(resources, dataSerializer));
      links.add(nodeId, linkArray);
    });
    json.add("nodes", nodeMap);
    json.add("layers", layerMap);
    json.add("links", links);
    final JsonObject labels = new JsonObject();
    this.labels.forEach((k, v) -> {
      labels.addProperty(k.toString(), v.toString());
    });
    json.add("labels", labels);
    json.addProperty("head", getHead().getId().toString());
    return json;
  }
  
  /**
   * Gets layer.
   *
   * @return the layer
   */
  public NNLayer getLayer() {
    return this;
  }
  
  private DAGNode getNode(final UUID id) {
    DAGNode returnValue = nodesById.get(id);
    if (null == returnValue) {
      returnValue = inputNodes.get(id);
    }
    return returnValue;
  }
  
  /**
   * Gets nodes.
   *
   * @return the nodes
   */
  public List<DAGNode> getNodes() {
    return Stream.concat(
      nodesById.values().stream(),
      getInput().stream()
                        ).collect(Collectors.toList());
  }
  
  private void initLinks(final Map<UUID, List<UUID>> nodeLinks, final Map<UUID, NNLayer> layersByNodeId, final UUID newNodeId) {
    if (layersById.containsKey(newNodeId)) return;
    if (inputNodes.containsKey(newNodeId)) return;
    final NNLayer layer = layersByNodeId.get(newNodeId);
    if (layer == null) {
      throw new IllegalArgumentException(String.format("%s is linked to but not defined", newNodeId));
    }
    final List<UUID> links = nodeLinks.get(newNodeId);
    if (null != links) {
      for (final UUID link : links) {
        initLinks(nodeLinks, layersByNodeId, link);
      }
    }
    assertConsistent();
    final DAGNode[] dependencies = getDependencies(nodeLinks, newNodeId);
    final InnerNode node = new InnerNode(this, layer, newNodeId, dependencies);
    layersById.put(layer.getId(), layer);
    nodesById.put(node.getId(), node);
    assertConsistent();
  }
  
  /**
   * Remove last input nn layer.
   *
   * @return the nn layer
   */
  public NNLayer removeLastInput() {
    final int index = inputHandles.size() - 1;
    final UUID key = inputHandles.remove(index);
    inputNodes.remove(key);
    return this;
  }
  
  /**
   * Reset.
   */
  public void reset() {
    layersById.clear();
    nodesById.clear();
    labels.clear();
  }
  
  @Override
  public DAGNetwork setFrozen(final boolean frozen) {
    super.setFrozen(frozen);
    if (null != layersById) {
      visitLayers(layer -> layer.setFrozen(frozen));
    }
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }
  
  /**
   * Visit layers.
   *
   * @param visitor the visitor
   */
  public void visitLayers(final Consumer<NNLayer> visitor) {
    layersById.values().forEach(layer -> {
      if (layer instanceof DAGNetwork) {
        ((DAGNetwork) layer).visitLayers(visitor);
      }
      if (layer instanceof WrapperLayer) {
        visitor.accept(((WrapperLayer) layer).getInner());
      }
      visitor.accept(layer);
    });
  }
  
  /**
   * Visit nodes.
   *
   * @param visitor the visitor
   */
  public void visitNodes(final Consumer<DAGNode> visitor) {
    nodesById.values().forEach(node -> {
      if (node.getLayer() instanceof DAGNetwork) {
        ((DAGNetwork) node.getLayer()).visitNodes(visitor);
      }
      visitor.accept(node);
    });
  }
  
}
