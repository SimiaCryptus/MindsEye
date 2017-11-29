# DropoutNoiseLayer
## DropoutNoiseLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.DropoutNoiseLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002ec",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/c88cbdf1-1c2a-4a5e-b964-8909000002ec",
      "inputs": [
        "187f91b4-b244-43d1-bd1b-0afd342b0ef8"
      ],
      "nodes": {
        "55a25f49-c2c8-44ad-8b0f-1b772f71492b": "c88cbdf1-1c2a-4a5e-b964-8909000002ee",
        "d3d6be82-c761-41d4-a26d-c817195efb6d": "c88cbdf1-1c2a-4a5e-b964-8909000002ed"
      },
      "layers": {
        "c88cbdf1-1c2a-4a5e-b964-8909000002ee": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-8909000002ee",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "c88cbdf1-1c2a-4a5e-b964-8909000002ed": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "c88cbdf1-1c2a-4a5e-b964-8909000002ed",
          "isFrozen": false,
          "name": "ProductInputsLayer/c88cbdf1-1c2a-4a5e-b964-8909000002ed"
        }
      },
      "links": {
        "55a25f49-c2c8-44ad-8b0f-1b772f71492b": [
          "187f91b4-b244-43d1-bd1b-0afd342b0ef8"
        ],
        "d3d6be82-c761-41d4-a26d-c817195efb6d": [
          "55a25f49-c2c8-44ad-8b0f-1b772f71492b",
          "187f91b4-b244-43d1-bd1b-0afd342b0ef8"
        ]
      },
      "labels": {},
      "head": "d3d6be82-c761-41d4-a26d-c817195efb6d"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 2.00 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.648, -1.668 ], [ -1.028, -1.748 ], [ 1.492, -0.508 ] ],
    	[ [ 1.36, 1.008 ], [ 0.556, -0.388 ], [ -1.276, -0.32 ] ],
    	[ [ -0.288, 0.732 ], [ -1.152, 1.448 ], [ 1.328, 1.272 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.6480000019073486, -1.6679999828338623 ], [ -1.027999997138977, -1.7480000257492065 ], [ 0.0, 0.0 ] ],
    	[ [ 1.3600000143051147, 1.0080000162124634 ], [ 0.0, -0.3880000114440918 ], [ 0.0, 0.0 ] ],
    	[ [ -0.2879999876022339, 0.0 ], [ 0.0, 1.4479999542236328 ], [ 0.0, 0.0 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.05 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: DropoutNoiseLayer/c88cbdf1-1c2a-4a5e-b964-8909000002ec
    Inputs: [
    	[ [ 1.648, -1.668 ], [ -1.028, -1.748 ], [ 1.492, -0.508 ] ],
    	[ [ 1.36, 1.008 ], [ 0.556, -0.388 ], [ -1.276, -0.32 ] ],
    	[ [ -0.288, 0.732 ], [ -1.152, 1.448 ], [ 1.328, 1.272 ] ]
    ]
    output=[
    	[ [ 1.6480000019073486, -1.6679999828338623 ], [ -1.027999997138977, -1.7480000257492065 ], [ 0.0, 0.0 ] ],
    	[ [ 1.3600000143051147, 1.0080000162124634 ], [ 0.0, -0.3880000114440918 ], [ 0.0, 0.0 ] ],
    	[ [ -0.2879999876022339, 0.0 ], [ 0.0, 1.4479999542236328 ], [ 0.0, 0.0 ] ]
    ]
    measured/actual: [ [ 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.9998679161071777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547 ] ]
    implemented/expected: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    error: [ [ 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.3208389282226562E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 4.5049e-06 +- 2.6710e-05 [0.0000e+00 - 1.6594e-04] (324#)
    relativeTol: 8.1083e-05 +- 5.3163e-06 [6.6046e-05 - 8.2963e-05] (9#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.07 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.4783 +- 0.8133 [2.6247 - 7.9367]
    Learning performance: 0.5381 +- 0.1313 [0.2992 - 1.2454]
    
```

