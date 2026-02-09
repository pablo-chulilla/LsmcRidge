import flet as ft
import numpy as np
import numpy.linalg as lng

def listToString(liste):
    ret = ""
    for i in range(len(liste)):
        ret = ret + str(liste[i])
        if i < len(liste) - 1:
            ret = ret + ", "
    return ret

def ridge(X, Y, lbd):
    return lng.solve(X.T@X + lbd*np.eye(X.shape[1]), X.T@Y)

def ols(X,Y):
    return lng.solve(X.T@X, X.T@Y)

def lsmc(X, Y, C, d):
    print("lsmc calcule")
    betaOls = ols(X, Y)
    XTXinv= lng.inv(X.T@X)
    cxtxinvctinv = lng.inv(C@XTXinv@C.T)
    return betaOls - XTXinv@C.T@cxtxinvctinv@(C@betaOls - d)

def ridgeContraints(X, Y, C, d, lbd):
    betaridge = ridge(X, Y, lbd)
    invterm = lng.inv(X.T@X + lbd*np.eye(X.shape[1]))
    invterm2 = lng.inv(C@invterm@C.T)
    return betaridge - invterm@C.T@invterm2@(C@betaridge - d)

def main(page: ft.Page):
    page.title = "Linear Regression Tool"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    
    x_inputs = []
    y_inputs = []
    x_constraints = []
    y_constraints = []
    rows_container = ft.Column()

    def add_row(e=None):
        # Create unique input fields for each row
        x_field = ft.TextField(label="X Value", width=150, border_radius=10)
        y_field = ft.TextField(label="Y Value", width=150, border_radius=10)
        x_inputs.append(x_field)
        y_inputs.append(y_field)
        
        rows_container.controls.append(
            ft.Row([x_field, y_field], alignment=ft.MainAxisAlignment.CENTER)
        )
        page.update()

    def add_constraint(e=None):
        x_field = ft.TextField(label="X Constraint Value", width = 150, border_radius=10)
        y_field = ft.TextField(label="Y Constraint Value", width = 150, border_radius=10)
        x_constraints.append(x_field)
        y_constraints.append(y_field)

        rows_container.controls.append(
            ft.Row([x_field, y_field], alignment = ft.MainAxisAlignment.END)
        )
        page.update()

    def restart(e):
        x_inputs.clear()
        y_inputs.clear()
        rows_container.controls.clear()


        input_section.visible = True
        config_section.visible = False
        add_row()
        page.update()

    # --- Math & Calculation ---
    def calculate(e):
        try:
            # Collect data from fields
            X_vals = [[float(x) for x in f.value.split(",")] for f in x_inputs]
            Y_vals = [[float(f.value)] for f in y_inputs]

            X_constraints = [[float(x) for x in f.value.split(",")] for f in x_constraints]
            Y_constraints = [[float(f.value)] for f in y_constraints]

            result_text.value=""
            
            if len(X_vals) < 2:
                result_text.value = "Error: Please provide at least 2 points."
                page.update()
                return
            X = np.array(X_vals)
            Y = np.array(Y_vals)
            Xconstraints = np.array(X_constraints)
            Yconstraints = np.array(Y_constraints)
            
            beta = None
            if method_dropdown.value == "LSMC":
                # Beta = (X^T * X)^-1 * X^T * Y
                beta = lsmc(X, Y, Xconstraints, Yconstraints)
            else:
                # Ridge: Beta = (X^T * X + lambda * I)^-1 * X^T * Y
                lbd = float(lambda_field.value)
                beta = ridgeContraints(X, Y, Xconstraints, Yconstraints, lbd)
            print(listToString(beta.transpose().flatten().tolist()))
            result_text.value = listToString(beta.transpose().flatten().tolist())
            print("pas d'erreur de string :()")
            result_text.color = ft.Colors.GREEN_700
        except Exception as ex:
            result_text.value = f"Error: {str(ex)}"
            result_text.color = ft.Colors.RED
        page.update()

    # --- UI Layout ---
    method_dropdown = ft.Dropdown(
        label="Method",
        width=200,
        options=[ft.dropdown.Option("LSMC"), ft.dropdown.Option("Ridge")],
        on_change=lambda e: (setattr(lambda_field, "visible", method_dropdown.value == "Ridge"), page.update())
    )
    lambda_field = ft.TextField(label="Lambda (λ)", width=120, value="1.0", visible=False)
    result_text = ft.Text(size=18, weight=ft.FontWeight.BOLD, selectable=True)

    config_section = ft.Column(
        visible=False,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            ft.Text("Step 2: Model Config", size=24, weight=ft.FontWeight.BOLD),
            method_dropdown,
            lambda_field,
            ft.ElevatedButton("Run Analysis", on_click=calculate, icon=ft.icons.PLAY_ARROW_ROUNDED),
            result_text
        ]
    )

    input_section = ft.Column(
        controls=[
            ft.Text("Step 1: Enter Data", size=24, weight=ft.FontWeight.BOLD),
            rows_container,
            ft.Row([
                ft.ElevatedButton("Add Row", icon=ft.icons.ADD, on_click=add_row),
                ft.ElevatedButton("Add Constraint", icon=ft.icons.ADD, on_click=add_constraint),
                ft.FilledButton("Next", icon=ft.icons.CHEVRON_RIGHT, on_click=lambda _: (
                    setattr(input_section, "visible", False),
                    setattr(config_section, "visible", True),
                    page.update()
                ))
            ], alignment=ft.MainAxisAlignment.CENTER)
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    page.add(
        ft.Row([ft.IconButton(ft.icons.RESTART_ALT, on_click=restart, tooltip="Restart")], alignment=ft.MainAxisAlignment.START),
        input_section,
        config_section
    )

    add_row()

if __name__ == "__main__":
    import os
    # The 'port' is dynamically assigned by web hosts
    port = int(os.getenv("PORT", 8080))
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=port)
