HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>MNIST 预测服务</title>
  <style>
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      max-width: 800px; 
      margin: 32px auto; 
      padding: 0 16px;
      background-color: #f5f5f5;
    }
    .card { 
      background: white;
      border: 1px solid #ddd; 
      padding: 24px; 
      border-radius: 8px; 
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }
    .row { margin: 16px 0; }
    label { display: block; margin-bottom: 8px; font-weight: 500; }
    select, input[type="file"] { 
      width: 100%; 
      padding: 8px; 
      border: 1px solid #ddd; 
      border-radius: 4px;
      font-size: 14px;
    }
    button { 
      background-color: #007bff; 
      color: white; 
      padding: 10px 24px; 
      border: none; 
      border-radius: 4px; 
      cursor: pointer;
      font-size: 16px;
      font-weight: 500;
    }
    button:hover { background-color: #0056b3; }
    button:disabled { background-color: #ccc; cursor: not-allowed; }
    #result { display: none; margin-top: 20px; }
    .prediction-box {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 24px;
      border-radius: 8px;
      text-align: center;
      margin-bottom: 20px;
    }
    .prediction-number {
      font-size: 72px;
      font-weight: bold;
      margin: 10px 0;
    }
    .confidence {
      font-size: 18px;
      opacity: 0.9;
    }
    .top-predictions {
      margin-top: 20px;
    }
    .top-predictions h3 {
      margin-bottom: 12px;
      color: #333;
    }
    .pred-item {
      display: flex;
      justify-content: space-between;
      padding: 12px;
      margin: 8px 0;
      background: #f8f9fa;
      border-radius: 4px;
      border-left: 4px solid #007bff;
    }
    .pred-digit {
      font-size: 24px;
      font-weight: bold;
      color: #333;
    }
    .pred-prob {
      color: #666;
      font-size: 14px;
    }
    .loading {
      text-align: center;
      color: #666;
      padding: 20px;
    }
    .error {
      background-color: #f8d7da;
      color: #721c24;
      padding: 12px;
      border-radius: 4px;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <h2>MNIST 手写数字预测服务</h2>
  <div class="card">
    <form id="predictForm" enctype="multipart/form-data">
      <div class="row">
        <label>选择模型：</label>
        <select name="model" id="modelSelect">
          <option value="cnn">CNN (卷积神经网络)</option>
          <option value="mlp">MLP (多层感知机)</option>
        </select>
      </div>
      <div class="row">
        <label>上传图片 (png/jpg/jpeg)：</label>
        <input type="file" name="file" id="fileInput" accept="image/*" required />
      </div>
      <div class="row">
        <button type="submit" id="submitBtn">开始预测</button>
      </div>
    </form>
    
    <div id="result">
      <div class="prediction-box">
        <div style="font-size: 18px; opacity: 0.9;">预测结果</div>
        <div class="prediction-number" id="predNumber">-</div>
        <div class="confidence" id="confidence">置信度: -</div>
      </div>
      
      <div class="top-predictions">
        <h3>Top 3 预测结果</h3>
        <div id="topPredictions"></div>
      </div>
    </div>
    
    <div id="loading" class="loading" style="display: none;">
      正在处理中，请稍候...
    </div>
    
    <div id="error" class="error" style="display: none;"></div>
  </div>

  <script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const formData = new FormData();
      formData.append('model', document.getElementById('modelSelect').value);
      formData.append('file', document.getElementById('fileInput').files[0]);
      
      const submitBtn = document.getElementById('submitBtn');
      const resultDiv = document.getElementById('result');
      const loadingDiv = document.getElementById('loading');
      const errorDiv = document.getElementById('error');
      
      submitBtn.disabled = true;
      resultDiv.style.display = 'none';
      errorDiv.style.display = 'none';
      loadingDiv.style.display = 'block';
      
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
          document.getElementById('predNumber').textContent = data.result.prediction;
          document.getElementById('confidence').textContent = '置信度: ' + data.result.confidence;
          
          const topPredDiv = document.getElementById('topPredictions');
          topPredDiv.innerHTML = '';
          data.result.top_predictions.forEach((pred, index) => {
            const predItem = document.createElement('div');
            predItem.className = 'pred-item';
            predItem.innerHTML = `
              <div>
                <span class="pred-digit">${pred.digit}</span>
                <span style="margin-left: 12px; color: #999;">第 ${index + 1} 可能</span>
              </div>
              <div class="pred-prob">${pred.confidence}</div>
            `;
            topPredDiv.appendChild(predItem);
          });
          
          resultDiv.style.display = 'block';
        } else {
          throw new Error(data.detail || '预测失败');
        }
      } catch (error) {
        errorDiv.textContent = '错误: ' + error.message;
        errorDiv.style.display = 'block';
      } finally {
        submitBtn.disabled = false;
        loadingDiv.style.display = 'none';
      }
    });
  </script>
</body>
</html>
"""

