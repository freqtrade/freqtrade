## 2026-02-01 - [FastAPI Validation & Exception Handling]
**Vulnerability:** Not strictly a vulnerability, but a pattern: When replacing explicit type declarations in FastAPI endpoints (e.g. `pair: str`) with custom Dependencies (e.g. `pair: str = Depends(validate)`), if the dependency makes the field optional (returns None) but the Response Model requires it, it causes a `ResponseValidationError` (500 error) instead of `RequestValidationError` (422 error).
**Learning:** Using `Query(..., pattern=r"...")` directly in the endpoint signature is safer and cleaner than custom Dependencies for simple validation, as it preserves the "required" nature of the field at the interface level and correctly triggers 422 for client errors.
**Prevention:** Prefer standard FastAPI validation (Pydantic/Query/Path) over custom dependencies for basic type/format checks to ensure correct error status codes.

## 2026-02-01 - [Pydantic Field Validation for List Sanitization]
**Vulnerability:** List inputs (e.g. `list[str]`) in API payloads can be used to inject malicious values (e.g. path traversal sequences) if not validated individually.
**Learning:** Pydantic's `field_validator` allows enforcing regex patterns on list elements to ensure they adhere to expected formats (e.g. pair names) before they reach business logic.
**Prevention:** Always add validators for list fields that are used in sensitive operations like file handling to reject malformed inputs early.
