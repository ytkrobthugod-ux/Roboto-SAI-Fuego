import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AuthForm } from '@/components/auth/AuthForm';
import { useAuthStore } from '@/stores/authStore';

function LocationDisplay() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

describe('AuthForm demo bypass', () => {
  beforeEach(() => {
    useAuthStore.setState(
      {
        userId: null,
        username: null,
        email: null,
        avatarUrl: null,
        provider: null,
        isLoggedIn: false,
      },
      false
    );
  });

  it('calls loginDemo and does not call onSubmit in demo mode', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();

    const loginDemoSpy = vi.spyOn(useAuthStore.getState(), 'loginDemo');

    render(
      <MemoryRouter initialEntries={['/login']}>
        <Routes>
          <Route
            path="/login"
            element={
              <>
                <AuthForm onSubmit={onSubmit} initialMode="login" />
                <LocationDisplay />
              </>
            }
          />
          <Route path="/chat" element={<LocationDisplay />} />
        </Routes>
      </MemoryRouter>
    );

    await user.click(screen.getByRole('button', { name: /demo login/i }));

    await user.type(screen.getByLabelText('Email'), 'demo@example.com');

    await user.click(screen.getByRole('button', { name: /enter demo/i }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(loginDemoSpy).toHaveBeenCalledWith('demo', 'demo@example.com');

    const loc = screen.getByTestId('location');
    expect(loc.textContent).toBe('/chat');
  });
});
