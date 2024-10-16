# Colab 실행 가이드
GPU 사용을 위해 Colab에서 실행해보겠습니다.  
아래 방법으로 손쉽게 소스코드들을 Colab에서 쉽게 열 수 있습니다.

---
### 1. google Colab에서 새 노트북 열기
![](assets/01_new_colab.png)
Colab : [https://colab.google](https://colab.google//)  
Colab에서 새 노트를 작성합니다.  

### 2. Google Drive 연결하기
![](assets/02_googledrive_mount1.png)  
현재 Colab 노트북은 임시 파일입니다. 드라이브에 마운트하여 파일을 저장할 수 있도록 해보겠습니다.    
왼쪽 네비게이션 바에서 구글드라이브 연결 아이콘을 클릭하고 안내에 따라 연결합니다.  
(세션이 연결되야 네비게이션 바가 나타납니다.)  
![](assets/03_googledrive_mount2.png)  
구글 드라이브가 연결되면 왼쪽에 폴더들이 생겨납니다.  
구글 드라이브는 `/content/drive/MyDrive`로 연결됩니다.  

### 3. Github Clone  
![](assets/04_gitclone1.png)  
[instuction-tuning-with-rag-example](https://github.com/aiqwe/instruction-tuning-with-rag-example.git) 레포지토리에서 주소를 복사합니다.
github의 https 주소를 복사합니다.  

![](assets/05_gitclone2.png)
노트북에서 `!git clone` 명령어와 함께 복사했던 github 주소와 함께 소스코드를 복사할 위치를 함께 입력합니다.  

![](assets/06_gitclone3.png)
아래와 같이 입력하면 구글드라이브에 `instruction-tuning-with-rage-example`이라는 폴더가 생기면서 소스코드가 해당 위치에 복사됩니다.
```
!git clone https://github.com/aiqwe/instruction-tuning-with-rag-example.git /content/drive/MyDrive/instruction-tuning-with-rag-example
```

### 4. 소스코드 Colab 노트북에서 실행하기  
![](assets/07_open_colab.png)
복사한 소스코드에서 `train.ipynb`을 우클릭하고 Google Colab에서 실행할 수 있습니다.

<br>
<br>
  
### etc. Colab에서 Secret 설정하기
![](assets/etc_secret.png)
왼쪽 세로 네게이션 바를 보면 위 그림과 같이 열쇠 그림이 있습니다. 여기서 Secret을 설정할 수 있습니다.  
Secret은 `google.colab.userdata.get()`를 통해 Key-Value(`dict` 타입같이)로 접근할 수 있습니다.
+ 노트북액세스 : Colab에서 접근 허용 여부입니다. 토글을 밀어서 ON으로 세팅합니다.
+ 이름 : Secret의 Key값입니다.
+ 값 : Secret의 Value입니다.
사진처럼 오른쪽 상단 빨간색 박스 안에는 사용하고 있는 GPU가 표시됩니다. 여기서 L4를 사용합니다. 
예제 코드는 `torch.bfloat16` 타입을 사용하기 때문에 Ampere 7 이전의 GPU에서는 지원되지 않습니다.  
Secret을 설정하는 방법은 [여기](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75)를 참조해 주세요.

### etc. import를 위한 path 지정
`utils.py`, `prompts.py`, `similarity.py` 모듈을 `import`하기 위해 해당 파이썬들이 위치한 폴더를 `sys.path.append`로 추가해야합니다.  
`drive.mount('/content/drive')` 로 구글 드라이브를 연결하고, `git clone`한 위치를 추가합니다.  
```python
sys.path.append("/content/drive/MyDrive/instruction-tuning-with-rag-example")
```
### etc. 허깅페이스 토큰
+ `train.ipynb` 에서 huggingface 토큰을 지정해야합니다.  
Gemma 모델을 로드하기 위해서는 huggingface의 gemma 모델 사용 신청을 하고, huggingface 토큰을 발급받아야 합니다.  
토큰 발급은 huggingface의 [User Access Tokens](https://huggingface.co/docs/hub/security-tokens)를 참조하세요.  
