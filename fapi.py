from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import requests
from obj_det.objd_util import detection_img, text_ocr, paddle_ocr
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from loguru import logger
import traceback
import config
import base64
from contextlib import asynccontextmanager


# 配置日志
logger.add("logs/fastapi_{time}.log", rotation="500 MB", retention="7 days", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("FastAPI OCR服务启动中...")
    yield
    logger.info("FastAPI OCR服务关闭中...")


app = FastAPI(
    title="OCR识别服务",
    description="提供多种OCR识别接口",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 常量配置
SECRET_KEY = '6aac5f82-141b-44a4-817f-369c64b12b19'
ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff', 'image/webp']


def img_decode(content: bytes) -> np.ndarray:
    """
    解码图片字节流为OpenCV图像数组
    
    Args:
        content: 图片字节流
        
    Returns:
        OpenCV图像数组
        
    Raises:
        ValueError: 如果图片解码失败
    """
    try:
        np_arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图片解码失败，可能不是有效的图片格式")
        return img
    except Exception as e:
        logger.error(f"图片解码错误: {str(e)}")
        raise ValueError(f"图片解码失败: {str(e)}")


async def verify_auth(secret: Optional[str] = Header(None, description="认证密钥")):
    """
    验证请求认证头
    
    Args:
        secret: 请求头中的secret值
        
    Returns:
        True 如果认证成功
        
    Raises:
        HTTPException: 如果认证失败
    """
    if not secret:
        logger.warning("请求缺少认证头")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证信息"
        )
    if secret != SECRET_KEY:
        logger.warning(f"非法访问尝试，错误的secret: {secret}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="非法访问，认证失败"
        )
    return True


class DataItem(BaseModel):
    """单个OCR数据项模型"""
    attachment_id: Optional[int] = Field(None, description="附件ID")
    data: Optional[str] = Field(None, description="图片数据，可以是base64字符串或URL")
    
    @validator('data')
    def validate_data(cls, v):
        if v and len(v) > 0:
            return v
        raise ValueError('data字段不能为空')


class RequestBodyModel(BaseModel):
    """批量OCR请求体模型"""
    data: Optional[List[DataItem]] = Field(None, description="待识别的图片数据列表")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "attachment_id": 1,
                        "data": "base64_encoded_image_string_or_url"
                    }
                ]
            }
        }


class OCRResultItem(BaseModel):
    """单个OCR结果模型"""
    attachment_id: Optional[int] = Field(None, description="附件ID")
    success: bool = Field(..., description="是否成功")
    message: str = Field("", description="错误信息")
    data: Dict[str, Any] = Field({}, description="OCR识别结果")


class ResponseModel(BaseModel):
    """统一响应模型"""
    result: Any = Field(..., description="响应结果")
    code: int = Field(..., description="响应状态码")


def get_base64_file(data) -> bytes:
    """
    获取并转换文件为base64编码
    
    Args:
        data: 可以是bytes、str(URL或base64字符串)
        
    Returns:
        base64编码的字节流
        
    Raises:
        ValueError: 如果数据格式不支持或解析失败
    """
    base64_str = b""
    
    try:
        if isinstance(data, bytes):
            # 如果是 UploadFile 读取的字节流
            base64_str = base64.b64encode(data)
            logger.debug("成功编码字节流为base64")
            
        elif isinstance(data, str):
            if data.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                # 从URL下载图片
                logger.info(f"从URL下载图片: {data}")
                response = requests.get(data, timeout=10)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image'):
                    raise ValueError(f'URL返回的内容不是图片类型: {content_type}')
                
                base64_str = base64.encodebytes(response.content)
                logger.debug(f"成功从URL获取图片并编码，大小: {len(response.content)} bytes")
            else:
                # 假设是base64字符串
                base64_str = data.encode("utf-8")
                logger.debug("使用提供的base64字符串")
        else:
            raise ValueError(f'不支持的数据类型: {type(data)}')
        
        if not base64_str:
            raise ValueError('解析图片失败，数据为空')
            
        return base64_str
        
    except requests.RequestException as e:
        logger.error(f"从URL下载图片失败: {str(e)}")
        raise ValueError(f'从URL下载图片失败: {str(e)}')
    except Exception as e:
        logger.error(f"解析图片数据失败: {str(e)}")
        raise ValueError(f'解析图片失败: {str(e)}')


@app.post(
    "/ocr",
    response_model=ResponseModel,
    summary="批量OCR识别",
    description="批量处理图片OCR识别，支持base64编码或URL",
    tags=["OCR识别"]
)
async def ocr(
    body: RequestBodyModel,
    auth: bool = Depends(verify_auth)
):
    """
    批量OCR识别接口
    
    - 支持批量处理多张图片
    - 图片数据可以是base64编码字符串或图片URL
    - 每张图片独立处理，失败不影响其他图片
    
    Returns:
        包含所有图片识别结果的响应
    """
    list_invoice = []
    
    try:
        if not body.data:
            logger.warning("请求body.data为空")
            return JSONResponse(
                content={'result': [], 'code': 400, 'message': '请求数据不能为空'},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"开始处理批量OCR，共 {len(body.data)} 张图片")
        
        for idx, d in enumerate(body.data):
            try:
                logger.info(f"处理第 {idx + 1}/{len(body.data)} 张图片, attachment_id: {d.attachment_id}")
                
                # 获取base64数据
                base64_data = get_base64_file(d.data)
                
                # 解码图片
                img = img_decode(base64.decodebytes(base64_data))
                
                # 执行OCR识别
                ocr_data = detection_img(img)
                
                list_invoice.append({
                    'attachment_id': d.attachment_id,
                    'success': True,
                    'message': "",
                    'data': ocr_data
                })
                
                logger.info(f"图片 {d.attachment_id} 识别成功")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"图片 {d.attachment_id} 识别失败: {error_msg}\n{traceback.format_exc()}")
                list_invoice.append({
                    'attachment_id': d.attachment_id,
                    'success': False,
                    'message': error_msg,
                    'data': {}
                })
        
        success_count = sum(1 for item in list_invoice if item['success'])
        logger.info(f"批量OCR处理完成，成功: {success_count}/{len(list_invoice)}")
        
        return JSONResponse(content={'result': list_invoice, 'code': 200})
        
    except Exception as e:
        logger.error(f"批量OCR处理异常: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            content={'result': list_invoice, 'code': 500, 'message': f'服务器错误: {str(e)}'},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.post(
    "/test_ocr",
    response_model=ResponseModel,
    summary="测试OCR识别",
    description="单张图片OCR测试接口，支持保存图片选项",
    tags=["OCR识别"]
)
async def test_ocr(
    file: UploadFile = File(..., description="待识别的图片文件"),
    saveImage: bool = Form(False, description="是否保存处理后的图片"),
    auth: bool = Depends(verify_auth)
):
    """
    测试OCR接口，支持文件上传
    
    - 支持单张图片上传识别
    - 可选择是否保存处理后的图片
    - 适用于接口测试和调试
    
    Args:
        file: 上传的图片文件
        saveImage: 是否保存处理后的图片，默认False
        
    Returns:
        OCR识别结果
    """
    list_invoice = []
    
    try:
        logger.info(f"收到测试OCR请求，文件名: {file.filename}, 类型: {file.content_type}, 保存图片: {saveImage}")
        
        # 验证文件类型
        if not file.content_type or file.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f'不支持的文件类型: {file.content_type}，请上传图片文件！')
        
        # 读取上传的文件
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"文件大小: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            raise ValueError('上传的文件为空')
        
        # 编码和解码
        base64_data = base64.b64encode(contents)
        img = img_decode(base64.decodebytes(base64_data))
        
        # 执行OCR识别
        logger.info("开始执行OCR识别...")
        ocr_data = detection_img(img, saveImage=saveImage)
        
        list_invoice.append({
            'attachment_id': 1,
            'success': True,
            'message': "",
            'data': ocr_data
        })
        
        logger.info(f"测试OCR识别成功，文件: {file.filename}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"测试OCR识别失败: {error_msg}\n{traceback.format_exc()}")
        list_invoice.append({
            'attachment_id': 1,
            'success': False,
            'message': error_msg,
            'data': {}
        })
    
    return JSONResponse(content={'result': list_invoice, 'code': 200})


@app.post(
    "/chineseOcr",
    response_model=ResponseModel,
    summary="中文OCR识别",
    description="专门用于中文文本识别的OCR接口",
    tags=["OCR识别"]
)
async def chinese_ocr(
    file: UploadFile = File(..., description="待识别的图片文件"),
    auth: bool = Depends(verify_auth)
):
    """
    中文OCR识别接口
    
    - 专门针对中文文本优化
    - 支持图片文件上传
    - 返回识别的文本结果
    
    Args:
        file: 上传的图片文件
        
    Returns:
        中文OCR识别结果
    """
    try:
        logger.info(f"收到中文OCR请求，文件名: {file.filename}, 类型: {file.content_type}")
        
        # 验证文件类型
        if not file.content_type or file.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f'不支持的文件类型: {file.content_type}，请上传图片文件！')
        
        # 读取上传的文件
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"文件大小: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            raise ValueError('上传的文件为空')
        
        # 编码和解码
        base64_data = base64.b64encode(contents)
        img = img_decode(base64.decodebytes(base64_data))
        
        # 执行中文OCR识别
        logger.info("开始执行中文OCR识别...")
        ocr_data = text_ocr(img)
        list_invoice = ocr_data
        
        logger.info(f"中文OCR识别成功，文件: {file.filename}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"中文OCR识别失败: {error_msg}\n{traceback.format_exc()}")
        list_invoice = f"识别错误: {error_msg}"
    
    return JSONResponse(content={'result': list_invoice, 'code': 200})


@app.post(
    "/paddle_ocr",
    response_model=ResponseModel,
    summary="PaddleOCR识别",
    description="使用PaddleOCR引擎进行文本识别",
    tags=["OCR识别"]
)
async def paddle_ocr_api(
    file: UploadFile = File(..., description="待识别的图片文件"),
    auth: bool = Depends(verify_auth)
):
    """
    PaddleOCR识别接口
    
    - 使用百度PaddleOCR引擎
    - 支持多语言文本识别
    - 高精度识别效果
    
    Args:
        file: 上传的图片文件
        
    Returns:
        PaddleOCR识别结果
    """
    try:
        logger.info(f"收到PaddleOCR请求，文件名: {file.filename}, 类型: {file.content_type}")
        
        # 验证文件类型
        if not file.content_type or file.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f'不支持的文件类型: {file.content_type}，请上传图片文件！')
        
        # 读取上传的文件
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"文件大小: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            raise ValueError('上传的文件为空')
        
        # 编码和解码
        base64_data = base64.b64encode(contents)
        img = img_decode(base64.decodebytes(base64_data))
        
        # 执行PaddleOCR识别
        logger.info("开始执行PaddleOCR识别...")
        ocr_data = paddle_ocr(img)
        list_invoice = ocr_data
        
        logger.info(f"PaddleOCR识别成功，文件: {file.filename}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"PaddleOCR识别失败: {error_msg}\n{traceback.format_exc()}")
        list_invoice = f"识别错误: {error_msg}"
    
    return JSONResponse(content={'result': list_invoice, 'code': 200})


@app.get(
    "/",
    summary="API根路径",
    description="返回API基本信息",
    tags=["系统"]
)
async def root():
    """
    根路径，返回API基本信息
    """
    return {
        "service": "OCR识别服务",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "批量OCR": "/ocr",
            "测试OCR": "/test_ocr",
            "中文OCR": "/chineseOcr",
            "PaddleOCR": "/paddle_ocr",
            "健康检查": "/health",
            "API文档": "/docs",
            "ReDoc文档": "/redoc"
        },
        "authentication": "需要在请求头中添加 secret 字段"
    }


@app.get(
    "/health",
    summary="健康检查",
    description="检查服务运行状态",
    tags=["系统"]
)
async def health_check():
    """
    健康检查接口
    
    - 用于监控服务运行状态
    - 无需认证
    
    Returns:
        服务健康状态
    """
    return {
        "status": "healthy",
        "service": "OCR识别服务",
        "version": "1.0.0"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTP异常处理器
    """
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "result": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    全局异常处理器
    """
    logger.error(f"未捕获的异常: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": 500,
            "message": f"服务器内部错误: {str(exc)}",
            "result": None
        }
    )


if __name__ == '__main__':
    import uvicorn
    
    # 服务器配置
    uvicorn_config = {
        "host": "0.0.0.0",
        "port": 8078,
        "reload": False,  # 生产环境设为False
        "workers": 1,  # 根据CPU核心数调整
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
    }
    
    logger.info(f"启动OCR FastAPI服务 - 端口: {uvicorn_config['port']}")
    uvicorn.run(app, **uvicorn_config)
