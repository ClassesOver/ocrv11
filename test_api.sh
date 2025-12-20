#!/bin/bash

# OCR API 测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
API_URL="${OCR_API_URL:-http://localhost:8078}"
API_SECRET="${OCR_API_SECRET:-6aac5f82-141b-44a4-817f-369c64b12b19}"

echo -e "${BLUE}======================================"
echo "    OCR API 测试"
echo "======================================${NC}"
echo "API 地址: $API_URL"
echo ""

# 检查服务是否运行
echo -e "${YELLOW}[1] 检查服务状态...${NC}"
if curl -s --max-time 5 "$API_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 服务正常运行${NC}"
else
    echo -e "${RED}✗ 服务未响应，请检查容器是否启动${NC}"
    echo "提示: docker ps | grep ocr-service"
    exit 1
fi

# 查找测试图片
echo -e "\n${YELLOW}[2] 查找测试图片...${NC}"
TEST_IMAGES=(
    "images/invoice/title.png"
    "images/invoice/invoice_code.png"
    "images/invoice/invoice_number.png"
)

FOUND_IMAGE=""
for img in "${TEST_IMAGES[@]}"; do
    if [ -f "$img" ]; then
        FOUND_IMAGE="$img"
        echo -e "${GREEN}✓ 找到测试图片: $img${NC}"
        break
    fi
done

if [ -z "$FOUND_IMAGE" ]; then
    echo -e "${RED}✗ 未找到测试图片${NC}"
    echo "请将图片放在以下位置之一:"
    for img in "${TEST_IMAGES[@]}"; do
        echo "  - $img"
    done
    exit 1
fi

# 测试 PaddleOCR 接口
echo -e "\n${YELLOW}[3] 测试 PaddleOCR 接口...${NC}"
echo "请求: POST $API_URL/paddle_ocr"

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/paddle_ocr" \
    -H "secret: $API_SECRET" \
    -F "file=@$FOUND_IMAGE")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP 状态码: $HTTP_CODE"

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ 请求成功${NC}"
    echo ""
    echo -e "${BLUE}响应内容:${NC}"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo -e "${RED}✗ 请求失败${NC}"
    echo "响应: $BODY"
    exit 1
fi

# 测试 OCR 接口
echo -e "\n${YELLOW}[4] 测试 OCR 接口...${NC}"
echo "请求: POST $API_URL/ocr"

# 将图片转为 base64
BASE64_DATA=$(base64 -w 0 "$FOUND_IMAGE" 2>/dev/null || base64 "$FOUND_IMAGE")

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/ocr" \
    -H "secret: $API_SECRET" \
    -H "Content-Type: application/json" \
    -d "{\"data\": [{\"data\": \"$BASE64_DATA\"}]}")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP 状态码: $HTTP_CODE"

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ 请求成功${NC}"
    echo ""
    echo -e "${BLUE}响应内容:${NC}"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo -e "${RED}✗ 请求失败${NC}"
    echo "响应: $BODY"
fi

# 性能测试
echo -e "\n${YELLOW}[5] 性能测试（10 次请求）...${NC}"

TOTAL_TIME=0
SUCCESS_COUNT=0

for i in {1..10}; do
    START=$(date +%s%N)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/paddle_ocr" \
        -H "secret: $API_SECRET" \
        -F "file=@$FOUND_IMAGE")
    END=$(date +%s%N)
    
    ELAPSED=$((($END - $START) / 1000000))
    TOTAL_TIME=$(($TOTAL_TIME + $ELAPSED))
    
    if [ "$HTTP_CODE" = "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "${GREEN}✓${NC} 请求 $i: ${ELAPSED}ms"
    else
        echo -e "${RED}✗${NC} 请求 $i: 失败 (HTTP $HTTP_CODE)"
    fi
done

AVG_TIME=$(($TOTAL_TIME / 10))

echo ""
echo -e "${BLUE}======================================"
echo "    测试总结"
echo "======================================${NC}"
echo "总请求数: 10"
echo "成功数: $SUCCESS_COUNT"
echo "失败数: $((10 - SUCCESS_COUNT))"
echo "平均响应时间: ${AVG_TIME}ms"
echo ""

if [ $SUCCESS_COUNT -eq 10 ]; then
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ 部分测试失败${NC}"
    exit 1
fi

