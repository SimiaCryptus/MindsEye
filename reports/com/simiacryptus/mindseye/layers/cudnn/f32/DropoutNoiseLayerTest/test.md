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
      "id": "f4569375-56fe-4e46-925c-95f4000000cf",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/f4569375-56fe-4e46-925c-95f4000000cf",
      "inputs": [
        "f7a31b0d-eec4-4178-a1e0-c245d9edbf04"
      ],
      "nodes": {
        "25d3d311-0bfb-451a-9d1c-670e67fa007f": "f4569375-56fe-4e46-925c-95f4000000d1",
        "fb91b6a9-a002-4d3f-99cc-8396dfe5e249": "f4569375-56fe-4e46-925c-95f4000000d0"
      },
      "layers": {
        "f4569375-56fe-4e46-925c-95f4000000d1": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "f4569375-56fe-4e46-925c-95f4000000d1",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "f4569375-56fe-4e46-925c-95f4000000d0": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "f4569375-56fe-4e46-925c-95f4000000d0",
          "isFrozen": false,
          "name": "ProductInputsLayer/f4569375-56fe-4e46-925c-95f4000000d0"
        }
      },
      "links": {
        "25d3d311-0bfb-451a-9d1c-670e67fa007f": [
          "f7a31b0d-eec4-4178-a1e0-c245d9edbf04"
        ],
        "fb91b6a9-a002-4d3f-99cc-8396dfe5e249": [
          "25d3d311-0bfb-451a-9d1c-670e67fa007f",
          "f7a31b0d-eec4-4178-a1e0-c245d9edbf04"
        ]
      },
      "labels": {},
      "head": "fb91b6a9-a002-4d3f-99cc-8396dfe5e249"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 2.01 seconds: 
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
    	[ [ -1.076, 0.136 ], [ -1.368, -1.164 ], [ -0.116, 0.368 ] ],
    	[ [ 1.752, -0.212 ], [ -0.732, 1.976 ], [ -1.06, -0.532 ] ],
    	[ [ -1.528, 0.34 ], [ -0.508, 1.424 ], [ 1.652, 1.924 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.13600000739097595 ], [ -1.3680000305175781, 0.0 ], [ -0.11599999666213989, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ -1.527999997138977, 0.3400000035762787 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -1.076, 0.136 ], [ -1.368, -1.164 ], [ -0.116, 0.368 ] ],
    	[ [ 1.752, -0.212 ], [ -0.732, 1.976 ], [ -1.06, -0.532 ] ],
    	[ [ -1.528, 0.34 ], [ -0.508, 1.424 ], [ 1.652, 1.924 ] ]
    ]
    Output: [
    	[ [ 0.0, 0.13600000739097595 ], [ -1.3680000305175781, 0.0 ], [ -0.11599999666213989, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ -1.527999997138977, 0.3400000035762787 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Measured: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999942421913147, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9998679161071777 ] ]
    Implemented: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.7578086853027344E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3208389282226562E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0174e-06 +- 1.6847e-05 [0.0000e+00 - 1.6594e-04] (324#)
    relativeTol: 6.5362e-05 +- 1.9789e-05 [2.8790e-05 - 8.2963e-05] (5#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.09 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.9936 +- 0.4222 [2.6389 - 4.7136]
    Learning performance: 0.5040 +- 0.0946 [0.3420 - 1.1000]
    
```

