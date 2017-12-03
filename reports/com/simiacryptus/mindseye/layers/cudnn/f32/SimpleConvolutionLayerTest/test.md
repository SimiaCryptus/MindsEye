# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00000394",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d00000394",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.764
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



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
    	[ [ -1.06 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.8698399066925049 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d00000395",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d00000395",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.764
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ -1.06 ] ]
    ]
    Error: [
    	[ [ -9.330749528579929E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 9.3307e-08 +- 0.0000e+00 [9.3307e-08 - 9.3307e-08] (1#)
    relativeTol: 2.4951e-08 +- 0.0000e+00 [2.4951e-08 - 2.4951e-08] (1#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -1.06 ] ]
    ]
    Output: [
    	[ [ 1.8698399066925049 ] ]
    ]
    Measured: [ [ -1.7619132995605469 ] ]
    Implemented: [ [ -1.7640000581741333 ] ]
    Error: [ [ 0.0020867586135864258 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ -1.06 ] ]
    ]
    Outputs: [
    	[ [ 1.8698399066925049 ] ]
    ]
    Measured Gradient: [ [ -1.0597705841064453 ] ]
    Implemented Gradient: [ [ -1.059999942779541 ] ]
    Error: [ [ 2.2935867309570312E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1581e-03 +- 9.2870e-04 [2.2936e-04 - 2.0868e-03] (2#)
    relativeTol: 3.5002e-04 +- 2.4182e-04 [1.0820e-04 - 5.9183e-04] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.16 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 5.7702 +- 0.6161 [5.0755 - 8.4496]
    Learning performance: 4.1454 +- 0.5627 [3.6449 - 8.7432]
    
```

