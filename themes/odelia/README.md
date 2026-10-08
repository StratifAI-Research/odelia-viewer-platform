# ODELIA login theme

A native Keycloak login theme matching the ODELIA viewer's dark OHIF styling. It supplies CSS,
English branding messages, the ODELIA logo, and a locally served Inter font. All templates and authentication
behavior are inherited from Keycloak's bundled keycloak theme.

## Activation

The platform Compose configuration mounts this directory read-only at /opt/keycloak/themes/odelia.
New installations select it through the ohif realm import. For existing installations, follow
the [login-theme configuration instructions](../../docs/setup/configuration.md#login-theme).

The login theme also styles inherited authentication screens such as password reset and profile
updates. Account-console and admin-console themes are configured separately.

## Customization

- login/resources/css/odelia.css: colors, layout, typography, focus and error states.
- login/messages/messages_en.properties: English page title, branding, heading and sign-in label.
- login/theme.properties: parent theme and stylesheet order.
- login/resources/fonts/: local font and its license.
- login/resources/img/odelia-logo-horizontal.jpg: the supplied horizontal ODELIA logo, unchanged.
  It is displayed on white to retain the original artwork colors; an accessible text label is retained.

Keep authentication templates inherited. If the parent Keycloak theme changes, check the CSS
selectors and test the complete sign-in flow before updating the pinned Keycloak image.

## Compatibility and verification

Targets the platform's pinned Keycloak 24.0.5 and its keycloak login parent. It is not based on the
newer keycloak.v2 login theme. Verify these states in an isolated realm when modifying styles:

- Desktop and mobile login, including focus indicators and password visibility.
- Invalid credentials and a successful OIDC callback.
- Password reset, required profile updates, and any other authentication steps enabled by the realm.
- Login with JavaScript disabled.

## Font and palette

The unmodified Inter Latin font is copied from the viewer 2.2.0 asset at
platform/ui-next/src/assets/woff2/latin.woff2. Its [SIL Open Font License](login/resources/fonts/OFL.txt)
is included beside it. Colors follow the OHIF Blue palette in that release's
platform/ui-next/src/tailwind.css. Font files are served by Keycloak; no external font service is used.

The logo is the project-provided ODE_Logo_horizontal_rgb.jpg asset. The stacked variant is not bundled.
