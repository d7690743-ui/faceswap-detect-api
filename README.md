# FaceSwap Detection API — Buffalo_L

API inicial para detectar rostros con InsightFace `buffalo_l`, pensada para conectar después con el frontend de FaceSwap.

## Endpoints

- `GET /` — información del servicio.
- `GET /health` — prueba sencilla de que la aplicación responde.
- `POST /detect` — recibe una imagen como `multipart/form-data`, con el campo `file`, y devuelve cajas, puntos faciales y embeddings compatibles con InsightFace.

## Importante sobre Vercel

Este repositorio es un **prototipo de API**, no una garantía de que `buffalo_l` vaya a funcionar en Vercel Serverless. El modelo y las dependencias nativas son pesados. La descarga/carga del modelo puede exceder los límites de tamaño, memoria o tiempo, especialmente en el primer arranque. La ruta `/health` no prueba que el modelo esté cargado.

Si el despliegue falla por tamaño, memoria o tiempo, conserva este código pero aloja la inferencia en un servicio con almacenamiento persistente y suficiente RAM (por ejemplo, una instancia de contenedor/CPU compatible), y deja Vercel como frontend o proxy. No intentes incluir pesos del modelo dentro de este ZIP.

## Despliegue básico

1. Descomprime la carpeta.
2. Sube su contenido a un repositorio de GitHub.
3. En Vercel, abre el proyecto `faceswap-detect-api` y conecta ese repositorio.
4. Despliega. Revisa los logs de compilación.
5. Prueba primero `/` y `/health`.
6. Prueba `/detect` con una imagen que tengas derecho a usar. Si la carga del modelo falla, revisa los logs y límites de Vercel.

## Probar `/detect`

Con `curl`:

```bash
curl -X POST "https://TU-DOMINIO.vercel.app/detect" \
  -F "file=@foto.jpg"
```

Respuesta aproximada:

```json
{
  "face_count": 1,
  "faces": [
    {
      "index": 0,
      "bbox": [100.0, 50.0, 300.0, 280.0],
      "det_score": 0.99,
      "landmarks": [[...]],
      "embedding": [...]
    }
  ]
}
```

Los números son solo ilustrativos.

## Seguridad y privacidad

- Limita `FRONTEND_ORIGIN` al dominio real del frontend cuando lo tengas.
- El endpoint devuelve embeddings faciales, que son datos biométricos sensibles. No los publiques ni los guardes sin consentimiento.
- Añade autenticación y límites de uso antes de exponer el endpoint públicamente.
- Usa únicamente imágenes propias o con permiso de las personas representadas.

## Licencias

Revisa las licencias vigentes de InsightFace y de cada peso/modelo descargado antes de usarlo comercialmente. El código de la API no otorga derechos adicionales sobre los modelos.
