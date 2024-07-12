from utils import *
import pandas as pd
from modal import Image, Secret, Mount, Volume, App, asgi_app
from fastapi import FastAPI, Response, Query, File, UploadFile, Request
from fastapi.responses import HTMLResponse
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Annotated
import os
import json
import re
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objects as go
import webbrowser
plt.style.use('seaborn-white')


image = (Image.micromamba()
         .pip_install("numpy==1.23.5", "pandas==1.5.3", "openpyxl==3.1.2", "fastapi==0.111.0",
                      "matplotlib==3.7.1", "seaborn==0.12.2", "python-dotenv==1.0.0", "PyQt5==5.15.10",
                      "google-cloud-aiplatform==1.48.0", "plotly==5.14.0",
                      )
         )

app = App(name="fastapi-genai-demo-v1", image=image)


class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response


genai_app = FastAPI(title='DataAnalystGenAIAPI',
                        summary="Data Analyst GenAi API", version="1.0",
                        contact={"name": "Cristian Vergara",
                                 "email": "cvergara@geti.cl"})

genai_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'])
genai_app.add_middleware(MyMiddleware)


@genai_app.get("/")
def read_root():
    return {"Root": "Root_test"}


@genai_app.get("/gemini-analyst")
async def generate_output(user_query: str,
                          type_of_plot: str = Query("plotly", enum=["plotly", "plt"])):
    try:
        # for now just loading a test xlsx
        file = 'df_test.xlsx'
        df = load_dataframe(f'{file}')
        print(df.head())
        response = analyze_table_gemini(query=user_query, df=df, plot_type=type_of_plot)
        print('Python Snippet Generated: \n')
        print(response)

        if 'python' in response:
            local_vars, output = execute_code(response, df=df)

        if type_of_plot == 'plotly':
            #if 'plt.' in response:
            if 'plotly.express' in response:
                try:
                    fig = local_vars['fig']
                    plot_html = fig.to_html(full_html=False)
                    return HTMLResponse(content=plot_html)

                except Exception as e:
                    try:
                        df_temp = local_vars['df_temp']
                        return df_temp.to_json(orient='records')

                    except Exception as e:
                        output = {
                            "error": str(e),
                            "output": "intente de nuevo probando con otra query"
                        }
                        return output

            else:
                try:
                    df_temp = local_vars['df_temp']
                    # Convert the DataFrame to JSON
                    return df_temp.to_json(orient='records')

                except Exception as e:

                    try:
                        output = {
                            "response": str(response),
                        }
                        return output

                    except Exception as e:

                        output = {
                            "error": str(e),
                            "output": "intente de nuevo probando con otra query"
                        }
                        return output

        elif type_of_plot == 'plt':
            if 'plt.' in response:
                try:
                    fig = local_vars['plt'].gcf()
                    output = io.BytesIO()
                    FigureCanvas(fig).print_png(output)
                    return Response(output.getvalue(), media_type="image/png")

                except Exception as e:
                    try:
                        df_temp = local_vars['df_temp']
                        return df_temp.to_json(orient='records')

                    except Exception as e:
                        output = {
                            "error": str(e),
                            "output": "intente de nuevo probando con otra query"
                        }
                        return output

            else:
                try:
                    df_temp = local_vars['df_temp']
                    # Convert the DataFrame to JSON
                    return df_temp.to_json(orient='records')

                except Exception as e:

                    try:
                        output = {
                            "response": str(response),
                        }
                        return output

                    except Exception as e:

                        output = {
                            "error": str(e),
                            "output": "intente de nuevo probando con otra query"
                        }
                        return output

    except Exception as e:
        output = {
            "error": str(e),
            "output": "intente de nuevo probando con otra query"
        }
        return output


@app.function(image=image,
              secret=Secret.from_name("automatch-secret-keys"),
              mounts=[Mount.from_local_file("key.json",
                      remote_path="/root/key.json"),
                      Mount.from_local_file("df_test.xlsx",
                                            remote_path="/root/df_test.xlsx")
                      ],
              _allow_background_volume_commits=True,
              timeout=999)  # schedule=Period(minutes=30)
@asgi_app(label='fastapi-genai-demo-v1')
def genai_fastapi_app():
    # check available GPUs
    print('####### GenAI v1.0 ####### \n'
          '#### test ####')

    return genai_app
