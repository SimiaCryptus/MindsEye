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
      "id": "0910987d-3688-428c-a892-e2c4000003fd",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/0910987d-3688-428c-a892-e2c4000003fd",
      "inputs": [
        "2558ab05-f25c-4dbc-9976-0cf2f9b70ac5"
      ],
      "nodes": {
        "455c5453-b573-45b8-9aae-5ce326dd1199": "0910987d-3688-428c-a892-e2c4000003ff",
        "2ced19d3-a64a-407f-8178-94aaa22d0dc8": "0910987d-3688-428c-a892-e2c4000003fe"
      },
      "layers": {
        "0910987d-3688-428c-a892-e2c4000003ff": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "0910987d-3688-428c-a892-e2c4000003ff",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "0910987d-3688-428c-a892-e2c4000003fe": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "0910987d-3688-428c-a892-e2c4000003fe",
          "isFrozen": false,
          "name": "ProductInputsLayer/0910987d-3688-428c-a892-e2c4000003fe"
        }
      },
      "links": {
        "455c5453-b573-45b8-9aae-5ce326dd1199": [
          "2558ab05-f25c-4dbc-9976-0cf2f9b70ac5"
        ],
        "2ced19d3-a64a-407f-8178-94aaa22d0dc8": [
          "455c5453-b573-45b8-9aae-5ce326dd1199",
          "2558ab05-f25c-4dbc-9976-0cf2f9b70ac5"
        ]
      },
      "labels": {},
      "head": "2ced19d3-a64a-407f-8178-94aaa22d0dc8"
    }
```



### Network Diagram
Code from [LayerTestBase.java:95](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L95) executed in 2.12 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -1.812, 1.188 ], [ 0.228, -1.4 ], [ -0.148, -0.528 ] ],
    	[ [ 1.108, -0.068 ], [ -1.364, 1.66 ], [ -0.204, -1.656 ] ],
    	[ [ -1.456, 0.8 ], [ -0.948, -0.248 ], [ 1.876, 0.856 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.812000036239624, 0.0 ], [ 0.2280000001192093, 0.0 ], [ 0.0, -0.527999997138977 ] ],
    	[ [ 1.1080000400543213, -0.06800000369548798 ], [ -1.3639999628067017, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ -1.4559999704360962, 0.0 ], [ -0.9480000138282776, 0.0 ], [ 1.8760000467300415, 0.8560000061988831 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.04 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.8567e-06 +- 6.1641e-05 [0.0000e+00 - 1.0262e-03] (324#)
    relativeTol: 1.1110e-04 +- 1.3726e-04 [8.4638e-06 - 5.1334e-04] (10#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.06 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.8551 +- 1.0946 [2.3739 - 10.9745]
    Learning performance: 0.5080 +- 0.1213 [0.3477 - 1.0573]
    
```

