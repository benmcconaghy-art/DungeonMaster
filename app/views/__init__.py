"""Server-side template-context builders.

Routes in :mod:`app.main` use these to compose the data their HTML
views need without inflating the route handlers themselves. Each
module here returns a plain ``dict`` ready for
``Jinja2Templates.TemplateResponse``.
"""
