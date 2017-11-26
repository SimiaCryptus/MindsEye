# ProductInputsLayer
## ProductInputsLayerTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
      "id": "b385277b-2d2d-42fe-8250-210c000000cf",
      "isFrozen": false,
      "name": "ProductInputsLayer/b385277b-2d2d-42fe-8250-210c000000cf"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.00 seconds: 
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
    	[ [ -1.832, -1.496 ], [ -1.612, -1.396 ] ],
    	[ [ -1.036, 1.396 ], [ -1.876, 1.136 ] ]
    ],
    [
    	[ [ 1.684, 0.82 ], [ -1.616, 1.628 ] ],
    	[ [ 1.972, -1.804 ], [ -0.204, 0.18 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.085088014602661, -1.2267199754714966 ], [ 2.604992151260376, -2.272688150405884 ] ],
    	[ [ -2.042992115020752, -2.5183839797973633 ], [ 0.3827039897441864, 0.20448002219200134 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:130](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L130) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.0888e-04 +- 3.7007e-04 [0.0000e+00 - 2.1058e-03] (128#)
    relativeTol: 3.0547e-04 +- 1.9778e-04 [1.6722e-05 - 6.4902e-04] (16#)
    
```

### Performance
Code from [LayerTestBase.java:135](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L135) executed in 4.87 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 2.4638 +- 0.3792 [2.0233 - 12.3054]
    Learning performance: 0.3705 +- 0.0831 [0.2052 - 1.6358]
    
```

